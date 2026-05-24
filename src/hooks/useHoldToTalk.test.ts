import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useHoldToTalk } from './useHoldToTalk';
import { HOLD_TO_TALK_MS, TAP_MAX_MS, COMMAND_WINDOW_MS } from '../config';

// jsdom doesn't implement Pointer Capture, but the hook calls it inside
// try/catch — these stubs prevent the catch from masking real failures.
function attachPointerCaptureStubs(): HTMLElement {
  const el = document.createElement('div');
  el.setPointerCapture = vi.fn();
  el.releasePointerCapture = vi.fn();
  return el;
}

function pointerEvent(
  type: 'pointerDown' | 'pointerUp' | 'pointerCancel' | 'pointerLeave',
  target: HTMLElement,
  opts: { pointerId?: number; pointerType?: string; button?: number } = {},
): ReactPointerEvent<HTMLElement> {
  return {
    pointerId: opts.pointerId ?? 1,
    pointerType: opts.pointerType ?? 'touch',
    button: opts.button ?? 0,
    currentTarget: target as unknown as ReactPointerEvent<HTMLElement>['currentTarget'],
    type,
  } as unknown as ReactPointerEvent<HTMLElement>;
}

function defaultOpts() {
  return {
    onTap: vi.fn(),
    onActivate: vi.fn(),
    onDeactivate: vi.fn(),
    unlockAudio: vi.fn(),
  };
}

describe('useHoldToTalk', () => {
  let now = 0;
  let perfSpy: ReturnType<typeof vi.spyOn>;
  let rafSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers();
    now = 1000;
    perfSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
    // requestAnimationFrame is a free-running loop in the hook; stub it as a
    // no-op so the tests don't churn. Progress values aren't asserted in
    // these tests; the timer-driven state changes are.
    rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 0);
  });

  afterEach(() => {
    perfSpy.mockRestore();
    rafSpy.mockRestore();
    vi.useRealTimers();
  });

  it('a short tap (< TAP_MAX_MS) fires onTap and not onActivate', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => {
      result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el));
    });
    // Release after 100 ms — well under TAP_MAX_MS (250).
    act(() => {
      now += 100;
      result.current.bindHandlers.onPointerUp(pointerEvent('pointerUp', el));
    });

    expect(opts.onTap).toHaveBeenCalledTimes(1);
    expect(opts.onActivate).not.toHaveBeenCalled();
    expect(opts.unlockAudio).toHaveBeenCalledTimes(1);
  });

  it('a release between TAP_MAX_MS and HOLD_TO_TALK_MS fires neither onTap nor onActivate', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => {
      result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el));
    });
    // Release after TAP_MAX_MS but before HOLD_TO_TALK_MS.
    act(() => {
      now += TAP_MAX_MS + 500;
      result.current.bindHandlers.onPointerUp(pointerEvent('pointerUp', el));
    });

    expect(opts.onTap).not.toHaveBeenCalled();
    expect(opts.onActivate).not.toHaveBeenCalled();
  });

  it('holding past HOLD_TO_TALK_MS fires onActivate exactly once and opens the command window', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => {
      result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el));
    });
    act(() => {
      now += HOLD_TO_TALK_MS;
      vi.advanceTimersByTime(HOLD_TO_TALK_MS);
    });

    expect(opts.onActivate).toHaveBeenCalledTimes(1);
    expect(result.current.isAwaitingCommand).toBe(true);
    expect(opts.onTap).not.toHaveBeenCalled();
  });

  it('does not fire onTap on the pointerup that follows a completed lock-in', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => { result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el)); });
    act(() => {
      now += HOLD_TO_TALK_MS;
      vi.advanceTimersByTime(HOLD_TO_TALK_MS);
    });
    act(() => {
      // User releases immediately after the lock-in beep.
      result.current.bindHandlers.onPointerUp(pointerEvent('pointerUp', el));
    });

    expect(opts.onTap).not.toHaveBeenCalled();
    // Window remains open (driven by command-window timer, not pointerup).
    expect(result.current.isAwaitingCommand).toBe(true);
  });

  it('pointercancel during the hold aborts without firing onActivate or onTap', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => { result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el)); });
    act(() => {
      now += 1500;
      result.current.bindHandlers.onPointerCancel(pointerEvent('pointerCancel', el));
    });
    // Advance past where the original lock-in would have fired.
    act(() => { vi.advanceTimersByTime(HOLD_TO_TALK_MS); });

    expect(opts.onActivate).not.toHaveBeenCalled();
    expect(opts.onTap).not.toHaveBeenCalled();
  });

  it('the command window auto-closes after COMMAND_WINDOW_MS', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => { result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el)); });
    act(() => {
      now += HOLD_TO_TALK_MS;
      vi.advanceTimersByTime(HOLD_TO_TALK_MS);
    });
    expect(result.current.isAwaitingCommand).toBe(true);

    act(() => {
      vi.advanceTimersByTime(COMMAND_WINDOW_MS);
    });
    expect(result.current.isAwaitingCommand).toBe(false);
    expect(opts.onDeactivate).toHaveBeenCalledTimes(1);
  });

  it('closeCommandWindow() lets the parent close early (when a command resolved)', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => { result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el)); });
    act(() => {
      now += HOLD_TO_TALK_MS;
      vi.advanceTimersByTime(HOLD_TO_TALK_MS);
    });
    expect(result.current.isAwaitingCommand).toBe(true);

    act(() => { result.current.closeCommandWindow(); });
    expect(result.current.isAwaitingCommand).toBe(false);
    expect(opts.onDeactivate).toHaveBeenCalledTimes(1);
  });

  it('disabled=true suppresses both tap and lock-in', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk({ ...opts, disabled: true }));
    const el = attachPointerCaptureStubs();

    act(() => { result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el)); });
    act(() => {
      now += HOLD_TO_TALK_MS;
      vi.advanceTimersByTime(HOLD_TO_TALK_MS);
    });
    act(() => {
      now += 100;
      result.current.bindHandlers.onPointerUp(pointerEvent('pointerUp', el));
    });

    expect(opts.onTap).not.toHaveBeenCalled();
    expect(opts.onActivate).not.toHaveBeenCalled();
  });

  it('ignores secondary pointers while one is active', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useHoldToTalk(opts));
    const el = attachPointerCaptureStubs();

    act(() => {
      result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el, { pointerId: 1 }));
    });
    // A second finger starts before the first is released.
    act(() => {
      now += 200;
      result.current.bindHandlers.onPointerDown(pointerEvent('pointerDown', el, { pointerId: 2 }));
    });
    expect(opts.unlockAudio).toHaveBeenCalledTimes(1);

    // Releasing the *second* pointer alone should not fire onTap — only the
    // originating pointerId should end the gesture.
    act(() => {
      result.current.bindHandlers.onPointerUp(pointerEvent('pointerUp', el, { pointerId: 2 }));
    });
    expect(opts.onTap).not.toHaveBeenCalled();
  });
});
