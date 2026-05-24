import { useCallback, useEffect, useRef, useState } from 'react';
import {
  HOLD_TO_TALK_MS,
  TAP_MAX_MS,
  COMMAND_WINDOW_MS,
} from '../config';

interface UseHoldToTalkOptions {
  // Fires for a pointerup BEFORE the hold threshold (quick tap).
  onTap: () => void;
  // Fires once when the hold passes HOLD_TO_TALK_MS — opens the command window.
  onActivate: () => void;
  // Fires when the command window closes (timeout OR external deactivate).
  // Used by App.tsx to deactivate SpeechRecognition cleanly if the user
  // never spoke during the window.
  onDeactivate: () => void;
  // First pointerdown counts as a user gesture — unlock audio here so the
  // beep + TTS work without a separate primer step.
  unlockAudio: () => void;
  // When true, the hold gesture is suppressed (e.g. Slow Lane is mid-call,
  // or the Settings dialog is open). Pointer events still fire visually
  // but produce no tap and no lock-in.
  disabled?: boolean;
}

type HoldHandlers = {
  onPointerDown: (event: React.PointerEvent<HTMLElement>) => void;
  onPointerUp: (event: React.PointerEvent<HTMLElement>) => void;
  onPointerCancel: (event: React.PointerEvent<HTMLElement>) => void;
  onPointerLeave: (event: React.PointerEvent<HTMLElement>) => void;
};

/**
 * Press-and-hold gesture for triggering on-demand voice commands.
 *
 * Lifecycle:
 *   pointerdown                        → start hold timer + progress rAF
 *   pointerup < TAP_MAX_MS             → onTap()  (quick describe)
 *   pointerup >= TAP_MAX_MS, < hold    → discarded (no tap, no command)
 *   hold passes HOLD_TO_TALK_MS (2 s)  → onActivate() + open command window
 *   command window closes              → onDeactivate()
 *
 * Only the first concurrent pointer is tracked. The hold can be cancelled
 * by pointercancel / pointerleave; this matches native iOS/Android touch
 * cancellation semantics (system gestures, multi-touch, finger off-screen).
 */
export function useHoldToTalk({
  onTap,
  onActivate,
  onDeactivate,
  unlockAudio,
  disabled = false,
}: UseHoldToTalkOptions) {
  // Public state mirrored from refs for rendering.
  const [holdProgress, setHoldProgress] = useState(0);
  const [isAwaitingCommand, setIsAwaitingCommand] = useState(false);

  // Refs survive re-renders and the timers don't need to trigger React updates.
  const activePointerIdRef = useRef<number | null>(null);
  const holdStartMsRef = useRef<number | null>(null);
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const windowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressRafRef = useRef<number | null>(null);
  const lockedInRef = useRef(false);

  // Keep the latest callbacks in a ref so the bound handlers don't have to
  // re-bind on every render. The .app-container would otherwise lose its
  // PointerEvent handlers between renders, which causes mid-gesture drops
  // on slow phones.
  const callbacksRef = useRef({ onTap, onActivate, onDeactivate, unlockAudio });
  useEffect(() => {
    callbacksRef.current = { onTap, onActivate, onDeactivate, unlockAudio };
  }, [onTap, onActivate, onDeactivate, unlockAudio]);

  const disabledRef = useRef(disabled);
  useEffect(() => { disabledRef.current = disabled; }, [disabled]);

  const clearHoldTimer = useCallback(() => {
    if (holdTimerRef.current !== null) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
  }, []);

  const clearWindowTimer = useCallback(() => {
    if (windowTimerRef.current !== null) {
      clearTimeout(windowTimerRef.current);
      windowTimerRef.current = null;
    }
  }, []);

  const stopProgressRaf = useCallback(() => {
    if (progressRafRef.current !== null) {
      cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = null;
    }
  }, []);

  const tickProgress = useCallback(() => {
    const start = holdStartMsRef.current;
    if (start === null) {
      setHoldProgress(0);
      return;
    }
    const elapsed = performance.now() - start;
    const p = Math.min(1, elapsed / HOLD_TO_TALK_MS);
    setHoldProgress(p);
    if (p < 1) {
      progressRafRef.current = requestAnimationFrame(tickProgress);
    }
  }, []);

  // ── Command window close (called when the timeout fires or the user
  // ── explicitly closes it via the returned closeCommandWindow function).
  const closeWindow = useCallback(() => {
    if (!lockedInRef.current) return;
    lockedInRef.current = false;
    clearWindowTimer();
    setIsAwaitingCommand(false);
    callbacksRef.current.onDeactivate();
  }, [clearWindowTimer]);

  // ── Lock-in: hold reached HOLD_TO_TALK_MS without being cancelled.
  const lockIn = useCallback(() => {
    if (disabledRef.current) {
      // Pointer was released or disabled while we were waiting — ignore.
      return;
    }
    lockedInRef.current = true;
    setHoldProgress(1);
    setIsAwaitingCommand(true);

    // Open the on-demand mic via the parent's onActivate callback.
    callbacksRef.current.onActivate();

    // Close the window after COMMAND_WINDOW_MS if no command arrives.
    windowTimerRef.current = setTimeout(() => {
      closeWindow();
    }, COMMAND_WINDOW_MS);
  }, [closeWindow]);

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (disabledRef.current) return;
    // Only honour primary mouse button or any touch/pen contact.
    if (event.pointerType === 'mouse' && event.button !== 0) return;

    // Ignore additional pointers while a gesture is in progress.
    if (activePointerIdRef.current !== null) return;
    // If the command window is already open, swallow further presses so
    // they don't accidentally fire onTap when the window closes.
    if (lockedInRef.current) return;

    // First pointerdown is the user gesture — unlock audio.
    callbacksRef.current.unlockAudio();

    activePointerIdRef.current = event.pointerId;
    holdStartMsRef.current = performance.now();
    setHoldProgress(0);

    try {
      // Pin the gesture to the originating element so dragging the finger
      // across the screen doesn't cancel mid-press (iOS Safari quirk).
      (event.currentTarget as Element | null)?.setPointerCapture?.(event.pointerId);
    } catch {
      // setPointerCapture isn't critical; older browsers may throw.
    }

    clearHoldTimer();
    holdTimerRef.current = setTimeout(lockIn, HOLD_TO_TALK_MS);

    stopProgressRaf();
    progressRafRef.current = requestAnimationFrame(tickProgress);
  }, [clearHoldTimer, lockIn, stopProgressRaf, tickProgress]);

  const finishGesture = useCallback((event: React.PointerEvent<HTMLElement>, fireTap: boolean) => {
    if (activePointerIdRef.current !== event.pointerId) return;

    const elapsed = holdStartMsRef.current === null
      ? 0
      : performance.now() - holdStartMsRef.current;

    clearHoldTimer();
    stopProgressRaf();
    setHoldProgress(0);
    activePointerIdRef.current = null;
    holdStartMsRef.current = null;

    try {
      (event.currentTarget as Element | null)?.releasePointerCapture?.(event.pointerId);
    } catch {
      // releasePointerCapture is best-effort.
    }

    // If we already locked in, the hold completed: do NOT fire onTap, even
    // if the user immediately released. The command window is now driving.
    if (lockedInRef.current) return;

    if (fireTap && elapsed < TAP_MAX_MS && !disabledRef.current) {
      callbacksRef.current.onTap();
    }
  }, [clearHoldTimer, stopProgressRaf]);

  const handlePointerUp = useCallback((event: React.PointerEvent<HTMLElement>) => {
    finishGesture(event, true);
  }, [finishGesture]);

  const handlePointerCancel = useCallback((event: React.PointerEvent<HTMLElement>) => {
    finishGesture(event, false);
  }, [finishGesture]);

  const handlePointerLeave = useCallback((event: React.PointerEvent<HTMLElement>) => {
    // pointerleave fires when pointerCapture isn't supported (older Android
    // WebView). With setPointerCapture, the event normally won't fire while
    // the pointer is still down. Treat it the same as cancel.
    finishGesture(event, false);
  }, [finishGesture]);

  // ── Cleanup on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      clearHoldTimer();
      clearWindowTimer();
      stopProgressRaf();
    };
  }, [clearHoldTimer, clearWindowTimer, stopProgressRaf]);

  // ── External close (called by App.tsx when a command resolves via voice
  // recognition, so the window doesn't sit open for the full 8 s).
  const closeCommandWindow = useCallback(() => {
    closeWindow();
  }, [closeWindow]);

  const handlers: HoldHandlers = {
    onPointerDown: handlePointerDown,
    onPointerUp: handlePointerUp,
    onPointerCancel: handlePointerCancel,
    onPointerLeave: handlePointerLeave,
  };

  return {
    bindHandlers: handlers,
    holdProgress,
    isAwaitingCommand,
    closeCommandWindow,
  };
}
