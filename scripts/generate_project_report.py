"""Generate a detailed VoiceEye project report as a styled DOCX file.

Run with the project's slides venv (python-docx is pre-installed there):

    .venv-slides/bin/python scripts/generate_project_report.py

Output: VoiceEye_Project_Report.docx (in repo root).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUT = Path("VoiceEye_Project_Report.docx")

# ── Brand palette ─────────────────────────────────────────────────────
INK = RGBColor(0x0F, 0x17, 0x2A)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
TEAL = RGBColor(0x0E, 0xA5, 0xA4)
TEAL_DEEP = RGBColor(0x0F, 0x76, 0x69)
INDIGO = RGBColor(0x4F, 0x46, 0xE5)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
RED = RGBColor(0xDC, 0x26, 0x26)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)

HEADER_BG_HEX = "0F172A"
ACCENT_BG_HEX = "E0F2FE"
TABLE_HEAD_HEX = "0EA5A4"
TABLE_ZEBRA_HEX = "F1F5F9"
CALLOUT_BG_HEX = "FEF3C7"


# ── Low-level XML helpers ─────────────────────────────────────────────
def _shade(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _set_cell_borders(cell, hex_color: str = "CBD5E1", size: int = 6) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), str(size))
        b.set(qn("w:color"), hex_color)
        tc_borders.append(b)
    tc_pr.append(tc_borders)


def _paragraph_shade(paragraph, hex_color: str) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    pPr.append(shd)


def _paragraph_border(paragraph, hex_color: str, size: int = 6, side: str = "left") -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement(f"w:{side}")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(size))
    b.set(qn("w:space"), "8")
    b.set(qn("w:color"), hex_color)
    pbdr.append(b)
    pPr.append(pbdr)


def _set_run(run, *, size_pt: float | None = None, bold: bool | None = None,
             color: RGBColor | None = None, font: str = "Aptos") -> None:
    run.font.name = font
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


# ── Document-level helpers ────────────────────────────────────────────
def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(11)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for level, size, color in [
        ("Heading 1", 22, INK),
        ("Heading 2", 16, TEAL_DEEP),
        ("Heading 3", 13, SLATE),
    ]:
        style = doc.styles[level]
        style.font.name = "Aptos Display" if level == "Heading 1" else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(18 if level == "Heading 1" else 10)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True


def set_page_margins(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)


def add_para(doc: Document, text: str = "", *, style: str | None = None,
             align: int | None = None, bold: bool = False,
             color: RGBColor | None = None, size: float | None = None,
             italic: bool = False):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if color is not None:
            run.font.color.rgb = color
        if size is not None:
            run.font.size = Pt(size)
    return p


def add_bullets(doc: Document, items: list[str], style: str = "List Bullet") -> None:
    for item in items:
        p = doc.add_paragraph(style=style)
        p.add_run(item)


def add_numbered(doc: Document, items: list[str]) -> None:
    add_bullets(doc, items, style="List Number")


def add_table(doc: Document, headers: list[str], rows: list[list[str]],
              col_widths: list[float] | None = None,
              header_fill: str = TABLE_HEAD_HEX,
              header_color: RGBColor = RGBColor(0xFF, 0xFF, 0xFF),
              zebra: str | None = TABLE_ZEBRA_HEX) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False

    if col_widths:
        for col, w in zip(table.columns, col_widths):
            for cell in col.cells:
                cell.width = Cm(w)

    # Header
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = ""
        _shade(cell, header_fill)
        _set_cell_borders(cell, "0F766E", 8)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        run = p.add_run(text)
        _set_run(run, size_pt=10.5, bold=True, color=header_color)

    # Body
    for r_i, row in enumerate(rows, start=1):
        tr = table.rows[r_i]
        fill = zebra if (zebra and r_i % 2 == 0) else None
        for cell, text in zip(tr.cells, row):
            cell.text = ""
            _set_cell_borders(cell, "E2E8F0", 4)
            if fill:
                _shade(cell, fill)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            p = cell.paragraphs[0]
            for i, line in enumerate(text.split("\n")):
                if i:
                    p = cell.add_paragraph()
                run = p.add_run(line)
                _set_run(run, size_pt=10, color=INK)
    doc.add_paragraph()


def add_callout(doc: Document, title: str, body: str, *,
                bg: str = CALLOUT_BG_HEX, accent: RGBColor = AMBER) -> None:
    table = doc.add_table(rows=1, cols=1)
    cell = table.rows[0].cells[0]
    _shade(cell, bg)
    _set_cell_borders(cell, "F59E0B", 6)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(title)
    _set_run(run, size_pt=11, bold=True, color=accent)
    body_p = cell.add_paragraph()
    body_run = body_p.add_run(body)
    _set_run(body_run, size_pt=10.5, color=INK)
    doc.add_paragraph()


def add_section_divider(doc: Document) -> None:
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "0EA5A4")
    pbdr.append(bottom)
    pPr.append(pbdr)


def add_cover(doc: Document) -> None:
    # Top accent block
    table = doc.add_table(rows=1, cols=1)
    cell = table.rows[0].cells[0]
    _shade(cell, HEADER_BG_HEX)
    _set_cell_borders(cell, "0F172A", 4)
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(36)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("VoiceEye")
    _set_run(run, size_pt=42, bold=True, color=RGBColor(0xF8, 0xFA, 0xFC), font="Aptos Display")

    sub = cell.add_paragraph()
    sub.paragraph_format.space_after = Pt(36)
    sub_run = sub.add_run("AI-powered real-time spatial awareness for visually impaired users")
    _set_run(sub_run, size_pt=14, bold=False, color=RGBColor(0x94, 0xA3, 0xB8))

    doc.add_paragraph()
    add_para(doc, "Detailed Project Report", bold=True, size=20, color=INK)
    add_para(doc, "Architecture, implementation, MLOps pipeline, and roadmap",
             size=12, color=MUTED)

    doc.add_paragraph()
    info_rows = [
        ["Project", "VoiceEye — assistive vision Progressive Web App"],
        ["Repository", "github.com/<user>/VoiceEye"],
        ["Live build", "https://voiceeye.pages.dev"],
        ["Document version", "1.0"],
        ["Generated", date.today().isoformat()],
        ["License", "MIT"],
    ]
    add_table(
        doc,
        headers=["Field", "Value"],
        rows=info_rows,
        col_widths=[5.0, 11.0],
        header_fill=TABLE_HEAD_HEX,
    )

    add_callout(
        doc,
        "Document scope",
        "This report describes VoiceEye end-to-end — the problem space, dual-lane architecture, "
        "browser runtime, voice and haptic UX, MLOps training pipeline, CI/CD, deployment story, "
        "testing strategy, known limitations, and the multi-phase implementation plan. It is "
        "intended as a single self-contained reference for engineers, reviewers, and stakeholders.",
    )

    doc.add_page_break()


def add_toc(doc: Document) -> None:
    add_para(doc, "Table of Contents", style="Heading 1")
    add_section_divider(doc)
    toc_items = [
        ("1.", "Executive Summary"),
        ("2.", "Problem Statement and Design Constraints"),
        ("3.", "System Architecture — Dual-Lane Processing"),
        ("4.", "Technology Stack"),
        ("5.", "Fast Lane — Detection, Tracking, Attention"),
        ("6.", "Slow Lane — Vision-Language Reasoning"),
        ("7.", "Voice, Audio, and Haptic Feedback"),
        ("8.", "Frontend Application Structure"),
        ("9.", "Settings, Error Handling, and Resilience"),
        ("10.", "MLOps and Training Pipeline"),
        ("11.", "Continuous Integration and Deployment"),
        ("12.", "Testing Strategy"),
        ("13.", "Performance, Memory, and Battery"),
        ("14.", "Privacy and Security Posture"),
        ("15.", "Known Limitations"),
        ("16.", "Implementation Plan and Roadmap"),
        ("17.", "Success Criteria"),
        ("18.", "Appendix A — Voice Commands"),
        ("19.", "Appendix B — Settings Reference"),
        ("20.", "Appendix C — File Inventory"),
        ("21.", "Appendix D — Glossary"),
    ]
    table = doc.add_table(rows=len(toc_items), cols=2)
    table.autofit = False
    for i, (num, title) in enumerate(toc_items):
        c0, c1 = table.rows[i].cells
        c0.width = Cm(2.0)
        c1.width = Cm(13.5)
        c0.text = ""
        c1.text = ""
        r0 = c0.paragraphs[0].add_run(num)
        _set_run(r0, size_pt=11, bold=True, color=TEAL_DEEP)
        r1 = c1.paragraphs[0].add_run(title)
        _set_run(r1, size_pt=11, color=INK)
    doc.add_page_break()


# ── Sections ──────────────────────────────────────────────────────────
def section_executive_summary(doc: Document) -> None:
    add_para(doc, "1. Executive Summary", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "VoiceEye is an accessibility-focused Progressive Web App (PWA) that turns a phone camera "
        "into a real-time spatial-awareness assistant for visually impaired users. The application "
        "is local-first: all object detection runs in the browser via ONNX Runtime Web, and the "
        "deeper vision-language reasoning runs against a local Ollama server in development or a "
        "Cloudflare Workers AI endpoint in production. No image data is ever sent to a third party.",
    )
    add_para(
        doc,
        "The system is organised around a Dual-Lane Processing architecture that decouples low-latency "
        "hazard detection from on-demand scene understanding. The Fast Lane runs continuously at "
        "approximately ten frames per second, applying YOLO26n inference, IoU-based tracking, "
        "monocular distance estimation, and a multi-stage attention pipeline that converts a stream "
        "of detections into a small, prioritised set of spoken announcements. The Slow Lane is "
        "triggered on demand by voice command or screen tap and produces scene descriptions, OCR "
        "readouts, or targeted object searches via a local Qwen3-VL model or a hosted Llama Vision "
        "model on the edge.",
    )
    add_para(
        doc,
        "VoiceEye is delivered as a static React + TypeScript bundle deployed to Cloudflare Pages. "
        "Training is reproducible via a ClearML pipeline with Optuna-based hyper-parameter optimisation "
        "and is gated by a quality threshold on mAP@50. The frontend ships with seventy-plus unit and "
        "integration tests, automated lint/build CI on every pull request, and a one-click deploy to "
        "the public production URL.",
    )

    add_para(doc, "Key facts at a glance", style="Heading 3")
    facts = [
        ["Architecture", "Dual-Lane Processing (Fast + Slow)"],
        ["Frontend", "React 19 + TypeScript 5.6 + Vite 6"],
        ["Detector", "YOLO26n via ONNX Runtime Web (WASM)"],
        ["Vision-language", "Qwen3-VL (Ollama) / Llama 3.2 Vision (Cloudflare Workers AI)"],
        ["Inference target", "≈ 10 FPS on mid-range mobile, p95 latency ≈ 145 ms"],
        ["Privacy posture", "On-device detection; VLM via local server or scoped edge function"],
        ["Tests", "70+ Vitest specs across utils, hooks, and voice recognition"],
        ["MLOps", "ClearML pipeline + Optuna HPO + quality gate (mAP@50 ≥ 0.40)"],
        ["Deployment", "Cloudflare Pages + Pages Functions, GitHub Actions CI/CD"],
    ]
    add_table(doc, ["Item", "Value"], facts, col_widths=[5.5, 10.5])


def section_problem(doc: Document) -> None:
    add_para(doc, "2. Problem Statement and Design Constraints", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "Visually impaired users navigate environments where context changes faster than any single "
        "scene description can be generated. Existing computer-vision aids tend to fall into two "
        "extremes: lightweight pedometer-style sensors that miss visual hazards, or full multimodal "
        "models whose latency is incompatible with avoiding an immediate obstacle. The product brief "
        "for VoiceEye is therefore three-fold: be fast enough for hazards, useful enough for context, "
        "and quiet enough for daily use without inducing alert fatigue.",
    )

    add_para(doc, "Constraints driving the design", style="Heading 3")
    add_bullets(doc, [
        "Low-latency requirement — hazard alerts must fire within roughly 100 ms of the frame in which the hazard becomes salient.",
        "Cognitive-load limit — a continuous narration of every detected object overwhelms users; announcements must be budgeted and prioritised.",
        "Privacy guarantee — image frames must never leave the device unless the user explicitly invokes the Slow Lane, and even then the destination must be a server the user controls or trusts.",
        "Mobile resource budget — sustained inference on a battery-powered phone must stay below thermal-throttling thresholds, with graceful degradation at low battery.",
        "Browser-only delivery — installation friction must be near zero, ruling out native app stores; the app is a PWA and runs over HTTPS with the camera and microphone APIs.",
        "Heterogeneous TTS / SR / haptic support — speech synthesis, speech recognition, and vibration APIs vary widely between browsers and OSes; the product must degrade gracefully when capabilities are missing.",
    ])

    add_callout(
        doc,
        "Design principle",
        "Every announcement consumes a finite cognitive budget. The system is built so that the most "
        "expensive thing it can do is interrupt the user with speech — therefore the attention pipeline "
        "is the central design artefact, not the detector itself.",
        bg="E0F2FE",
        accent=INDIGO,
    )


def section_architecture(doc: Document) -> None:
    add_para(doc, "3. System Architecture — Dual-Lane Processing", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "VoiceEye splits perception into two parallel lanes that share the camera stream but optimise "
        "for different latency budgets. The Fast Lane is always on; the Slow Lane fires on demand. "
        "Both lanes converge at the audio output layer, where the spatial-audio hook serialises and "
        "throttles speech, beeps, and haptics so the two lanes never collide.",
    )

    add_table(
        doc,
        headers=["Lane", "Engine", "Latency", "Cadence", "Purpose"],
        rows=[
            ["Fast", "YOLO26n + ONNX Runtime Web (WASM)", "≈ 100–150 ms / frame",
             "Continuous, throttled to ~10 FPS", "Object detection, tracking, hazard alerts"],
            ["Slow", "Qwen3-VL (Ollama) or Llama 3.2 Vision (Workers AI)", "≈ 2–8 s / call",
             "On-demand (voice or tap)", "Scene description, OCR, targeted object search"],
        ],
        col_widths=[2.0, 4.5, 3.0, 3.0, 3.5],
    )

    add_para(doc, "Pipeline overview", style="Heading 3")
    add_numbered(doc, [
        "Camera stream is acquired via getUserMedia with a back-facing preference and automatic retry on transient errors.",
        "Each rAF tick samples a frame; the Fast Lane runs YOLO inference at most every 100 ms.",
        "Detections feed an IoU tracker that assigns persistent track IDs and computes EMA-smoothed velocity.",
        "Monocular distance is estimated from bounding-box height and known average object heights.",
        "The attention pipeline filters tracks by tier, proximity, and budget into a small set of announcements.",
        "Announcements are joined into a single TTS utterance per frame and delivered alongside haptics and de-escalation tones.",
        "On user trigger, the Slow Lane captures the current frame as a 384 px JPEG, prompts the VLM, and speaks the response with timeout and error feedback.",
    ])

    add_callout(
        doc,
        "Why two lanes",
        "A single VLM cannot answer 'is there a car coming?' in under 100 ms; a single detector cannot "
        "answer 'what does the menu say?' Splitting concerns lets each engine be sized for its own job, "
        "and lets the user feel both safe and informed without either lane starving the other.",
        bg=ACCENT_BG_HEX,
        accent=INDIGO,
    )


def section_tech_stack(doc: Document) -> None:
    add_para(doc, "4. Technology Stack", style="Heading 1")
    add_section_divider(doc)
    add_para(doc, "Frontend and runtime", style="Heading 3")
    add_table(
        doc,
        ["Component", "Technology", "Notes"],
        [
            ["Framework", "React 19 + TypeScript 5.6", "Strict null checks, no implicit any"],
            ["Build tool", "Vite 6", "Static bundle, PWA plugin, HTTPS dev server"],
            ["Detection", "ONNX Runtime Web (WASM) + YOLO26n", "Float32, opset 17"],
            ["VLM (dev)", "Ollama + Qwen3-VL", "REST API on 127.0.0.1:11434, proxied by Vite"],
            ["VLM (prod)", "Cloudflare Workers AI (Llama 3.2 Vision)", "Pages Function on /api/vlm"],
            ["Voice input", "SpeechRecognition (Web Speech API)", "Wake word + command parser"],
            ["Voice output", "SpeechSynthesis", "Per-frame batched utterance"],
            ["Haptics", "navigator.vibrate", "Distinct patterns for proximity, collision, approach"],
            ["Styling", "Vanilla CSS — glassmorphism", "Backdrop blur, indigo primary palette"],
            ["Icons", "Lucide React", ""],
        ],
        col_widths=[3.5, 5.5, 7.0],
    )

    add_para(doc, "ML pipeline and ops", style="Heading 3")
    add_table(
        doc,
        ["Component", "Technology", "Notes"],
        [
            ["Model architecture", "YOLO26n", "2.57 M parameters, 6.1 GFLOPs"],
            ["Training framework", "Ultralytics YOLO (Python)", "AdamW, cosine LR, 150 epochs"],
            ["Experiment tracking", "ClearML", "Tasks, datasets, artifacts, agent queues"],
            ["HPO", "Optuna TPE via ClearML HyperParameterOptimizer", "20 trials, 8 h budget"],
            ["Quality gate", "mAP@50 ≥ 0.40", "Enforced in evaluation step"],
            ["CI/CD", "GitHub Actions", "Frontend lint+build, training-config validate, model download"],
            ["Deployment", "Cloudflare Pages + Pages Functions", "Edge static + serverless VLM proxy"],
        ],
        col_widths=[3.5, 5.5, 7.0],
    )


def section_fast_lane(doc: Document) -> None:
    add_para(doc, "5. Fast Lane — Detection, Tracking, Attention", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "The Fast Lane consists of three cooperating stages: a YOLO inference layer, an IoU-based "
        "multi-object tracker, and an attention pipeline that decides which tracks become announcements. "
        "Each stage is implemented in a dedicated module under src/utils/.",
    )

    add_para(doc, "5.1 YOLO inference (src/utils/yolo.ts)", style="Heading 2")
    add_bullets(doc, [
        "Loads an ONNX session lazily on first inference; the WASM binary is bundled by Vite from onnxruntime-web/wasm.",
        "Reuses a singleton OffscreenCanvas and a single pre-allocated Float32 input buffer, eliminating per-frame allocations.",
        "Letterboxes the camera frame to 640×640 with deterministic padding, then normalises to [0, 1].",
        "Decodes the [1, 84, 8400] output, applies confidence and IoU-based non-maximum suppression, and returns a list of class+box+score tuples.",
        "Disposes input and output tensors in try/finally blocks to prevent WASM heap leaks.",
        "Inference cadence is throttled to 100 ms via performance.now() inside the requestAnimationFrame loop.",
    ])

    add_para(doc, "5.2 IoU tracker (src/utils/tracker.ts)", style="Heading 2")
    add_bullets(doc, [
        "Each detection is matched to an existing track using IoU ≥ TRACKER_MIN_IOU = 0.3.",
        "Unmatched tracks survive up to TRACKER_MAX_AGE = 10 frames before being dropped; a re-entered object receives a new track ID.",
        "EMA-smoothed velocity (α = 0.3) tracks centroid translation and bounding-box area growth.",
        "Proximity zones are recomputed every frame from bbox-area / screen-area: safe (<25%), near (25–50%), danger (>50%).",
        "When the active track count would exceed MAX_TRACKS, eviction uses priority = tier_weight × zone_weight − age × EVICTION_AGE_PENALTY so tier-1 danger tracks are never evicted in favour of tier-3 safe ones.",
    ])

    add_para(doc, "5.3 Attention pipeline (src/utils/attention.ts)", style="Heading 2")
    add_para(
        doc,
        "The attention pipeline is the single most important user-experience component in the system. "
        "It receives the current track set, applies a sequence of priority and gating rules, and emits "
        "a small ranked list of announcements together with de-escalation cues and the boxes to render. "
        "The function signature is runAttention(tracks, now, state, config).",
    )
    add_table(
        doc,
        ["Field", "Meaning"],
        [
            ["toAnnounce", "Announcements that passed priority + budget + cooldown gating; consumed by speak() and haptics."],
            ["deescalationTones", "Track IDs that just exited the danger zone — each fires a 440 Hz / 50 ms tone (distinct from the 880 Hz wake beep)."],
            ["renderTracks", "Capped at MAX_RENDERED_BOXES = 8, sorted by priority, used for the on-screen overlay."],
            ["suppressedCount", "Number of tracks that were eligible but withheld; emitted as a metric only."],
        ],
        col_widths=[4.5, 12.0],
    )

    add_para(doc, "Gating rules", style="Heading 3")
    add_bullets(doc, [
        "Priority = tier_weight × zone_weight × (1 + APPROACHING_BOOST if areaGrowthRate exceeds threshold). Hazardous classes (car, bus, truck, motorcycle, bicycle) are tier 1; people are tier 2; everything else tier 3.",
        "Budget — at most VERBOSITY_K announcements per BUDGET_WINDOW_MS = 2000 ms: Quiet = 1, Normal = 3, Detailed = 6.",
        "Reason gating — zone-escalation (first-time entry to near/danger), approaching (growing bbox on a hazardous class), sustained (still in danger after SUSTAINED_INTERVAL_MS, capped by SUSTAINED_MAX_COUNT), or first-time (default single announcement).",
        "Tier-1 override — a tier-1 track making a safe → danger step-jump bypasses the budget.",
        "Spatial clustering — ≥ CLUSTER_MIN_MEMBERS same-class tracks within CLUSTER_MATCH_FRAC = 15 % of frame diagonal are announced as one group (e.g. \"3 people, ~4 m\") instead of individually.",
        "Cooldown garbage collection — entries older than COOLDOWN_GC_MS are dropped from the state map each frame to bound memory.",
    ])

    add_callout(
        doc,
        "Why batch into one utterance",
        "Because useSpatialAudio.speak() calls speechSynthesis.cancel(), the detection loop joins all "
        "per-frame announcements into a single utterance (\"A. B. C\"). Without this, the budget window's "
        "K > 1 would collapse to the last announcement only and the cluster + tier-1 override would lose their effect.",
        bg=CALLOUT_BG_HEX,
        accent=AMBER,
    )

    add_para(doc, "5.4 Distance estimation (src/utils/distance.ts)", style="Heading 2")
    add_para(
        doc,
        "Distance is estimated using a pinhole camera model. For each detected class, a known average "
        "real-world height (e.g. ≈ 1.7 m for an adult, ≈ 1.5 m for a sedan) and an assumed ≈ 70° "
        "vertical field of view are combined with the bounding-box height in pixels to produce a "
        "distance estimate. The result is rounded to a single decimal metre and clamped to >5 m for "
        "small detections to avoid spurious precision.",
    )

    add_para(doc, "5.5 Inference metrics (src/utils/inferenceMetrics.ts)", style="Heading 2")
    add_para(
        doc,
        "A singleton metrics object maintains a circular buffer of the last 100 frames and tracks "
        "inference latency, FPS, confidence distribution, and per-frame announcement and cluster "
        "counts. Every 300 frames it logs a summary line such as: "
        "[VoiceEye] fps=8.2 avgLatency=118ms p95=145ms detections/frame=3.1 announcements/min=… suppressed=… clusters=…",
    )


def section_slow_lane(doc: Document) -> None:
    add_para(doc, "6. Slow Lane — Vision-Language Reasoning", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "The Slow Lane is implemented in the useVLMEngine hook. It is responsible for capturing a "
        "single frame, building an appropriate prompt for one of three modes, dispatching to the "
        "configured VLM endpoint with timeout and deduplication, and routing the textual response "
        "back to the spatial-audio hook for spoken delivery.",
    )

    add_para(doc, "Trigger flow", style="Heading 3")
    add_numbered(doc, [
        "User triggers via wake word + command (e.g. \"Voice Eye, describe\") or a screen tap.",
        "An 880 Hz beep fires immediately via Web Audio API as instant confirmation; \"Got it\" is queued via speakQuick().",
        "useVLMEngine captures the current video frame, downscales to 384 px, and encodes as a base-64 JPEG.",
        "Mode is selected: Describe, Read (OCR), or Find <object>; each maps to a different prompt template.",
        "Request is dispatched with a 15-second AbortController timeout. Identical in-flight requests are deduplicated.",
        "Response text is spoken via the standard speak() path. On timeout or error, the user hears a clear spoken message instead of silence.",
    ])

    add_para(doc, "Endpoint selection", style="Heading 3")
    add_table(
        doc,
        ["Priority", "Endpoint", "When"],
        [
            ["1", "VITE_OLLAMA_URL", "Set explicitly — used with Cloudflare Tunnel for production parity testing"],
            ["2", "/api/ollama (Vite proxy → 127.0.0.1:11434)", "Local development with vite dev"],
            ["3", "/api/vlm (Cloudflare Pages Function → Workers AI)", "Production build (default)"],
        ],
        col_widths=[2.0, 7.5, 7.0],
    )

    add_para(doc, "Error paths", style="Heading 3")
    add_bullets(doc, [
        "Timeout (>15 s) — \"Sorry, that took too long. Please try again.\"",
        "Ollama unreachable — \"VLM server is not running. Please start Ollama and try again.\"",
        "Generic error — \"Something went wrong with the scene description.\"",
        "Origin restriction — Pages Function rejects any request whose Origin is not in VLM_ALLOWED_ORIGINS.",
    ])


def section_voice(doc: Document) -> None:
    add_para(doc, "7. Voice, Audio, and Haptic Feedback", style="Heading 1")
    add_section_divider(doc)

    add_para(doc, "Wake word and command parser", style="Heading 3")
    add_para(
        doc,
        "useVoiceRecognition holds a long-running SpeechRecognition session. A regex-based parser "
        "tolerates several pronunciations and whitespace variants of the wake phrase (\"Voice Eye\", "
        "\"Voiceeye\", \"Voice I\") and routes intents to one of three commands: describe, read, or "
        "find. The find command captures a noun phrase as a query string. Test coverage for this "
        "parser lives at src/hooks/useVoiceRecognition.test.ts.",
    )

    add_para(doc, "Spatial audio (useSpatialAudio.ts)", style="Heading 3")
    add_bullets(doc, [
        "speak(text) — primary TTS path; cancels prior utterance to avoid queue build-up.",
        "speakQuick(text) — non-cancelling utterance for short confirmations like \"Got it\".",
        "beep(freq, ms) — Web Audio API tone; 880 Hz on wake-word trigger, 440 Hz on danger-zone exit.",
        "vibrate(pattern) — wraps navigator.vibrate; gated by the Haptics setting.",
    ])

    add_para(doc, "Haptic patterns", style="Heading 3")
    add_table(
        doc,
        ["Pattern", "When", "Sequence (ms)"],
        [
            ["Single pulse", "Object occupies > 50 % of the screen", "[100]"],
            ["Triple pulse", "Collision warning ( > 60 %)", "[50, 50, 50]"],
            ["Rapid burst", "Hazardous class approaching", "[50, 30, 50, 30, 50]"],
        ],
        col_widths=[3.5, 7.5, 5.5],
    )

    add_callout(
        doc,
        "Voice UX detail",
        "On wake-word detection, the 880 Hz beep is fired *before* the TTS prompt resolves, because "
        "speech synthesis can take several hundred milliseconds to begin speaking on iOS Safari. The "
        "beep gives the user instant acoustic confirmation that the wake word was heard.",
        bg="DCFCE7",
        accent=GREEN,
    )


def section_app_structure(doc: Document) -> None:
    add_para(doc, "8. Frontend Application Structure", style="Heading 1")
    add_section_divider(doc)

    add_para(doc, "Source tree (src/)", style="Heading 3")
    add_table(
        doc,
        ["File", "Lines", "Responsibility"],
        [
            ["App.tsx", "213", "Slim orchestrator: wires hooks, tracks ref state, mediates lanes"],
            ["main.tsx", "—", "Entry point; mounts ErrorBoundary"],
            ["config.ts", "117", "All tunable constants — single source of truth"],
            ["index.css", "—", "Glassmorphism design system"],
            ["hooks/useDetectionLoop.ts", "158", "Fast Lane orchestration: rAF, YOLO, tracker, attention, alerts"],
            ["hooks/useVLMEngine.ts", "160", "Slow Lane: capture, prompt, dispatch, timeout, dedupe"],
            ["hooks/useVoiceRecognition.ts", "218", "Wake word + command parser; tests at *.test.ts"],
            ["hooks/useSpatialAudio.ts", "82", "TTS, beep, haptic primitives"],
            ["components/CameraView.tsx", "101", "Camera access + retry + spoken errors"],
            ["components/SettingsPanel.tsx", "137", "Settings UI + localStorage persistence"],
            ["components/ErrorBoundary.tsx", "105", "Crash recovery with TTS feedback"],
            ["utils/yolo.ts", "240", "ONNX session + pre/post-processing + NMS"],
            ["utils/tracker.ts", "158", "IoU tracker + velocity + proximity zones"],
            ["utils/attention.ts", "377", "Priority + budget + cluster + cooldown filter"],
            ["utils/distance.ts", "59", "Pinhole-camera distance estimator"],
            ["utils/inferenceMetrics.ts", "120", "Latency / FPS / confidence + announcement counters"],
        ],
        col_widths=[5.5, 1.5, 9.0],
    )

    add_para(doc, "State strategy", style="Heading 3")
    add_bullets(doc, [
        "tracksRef (mutable React ref) holds the live track list, avoiding stale closures inside the rAF loop.",
        "renderedTracks state is set from tracksRef each frame to drive React rendering of the bounding boxes.",
        "Settings are mirrored in settingsRef so the rAF loop sees the latest values without re-binding.",
        "All tunable thresholds live in src/config.ts — there are no magic numbers in feature code.",
    ])


def section_resilience(doc: Document) -> None:
    add_para(doc, "9. Settings, Error Handling, and Resilience", style="Heading 1")
    add_section_divider(doc)

    add_para(doc, "Settings persistence", style="Heading 3")
    add_para(
        doc,
        "All user-controllable settings are persisted to localStorage under the voiceeye_settings "
        "key. The SettingsPanel writes through on every change, and the useSpatialAudio and "
        "useDetectionLoop hooks read from a settings ref so updates take effect on the next frame.",
    )

    add_para(doc, "Error handling layers", style="Heading 3")
    add_table(
        doc,
        ["Layer", "Failure handled", "User feedback"],
        [
            ["CameraView", "DOMException from getUserMedia", "Spoken error + on-screen message + retry"],
            ["useVLMEngine", "Timeout, network, server unreachable", "Spoken error message, never silent"],
            ["useVoiceRecognition", "Recognition aborted by browser", "Automatic restart on recoverable errors"],
            ["ErrorBoundary", "Uncaught React error", "TTS \"VoiceEye encountered an error\", tap-to-reload"],
            ["Service worker", "Stale cache after a failed deploy", "Self-unregisters on first launch of the new build"],
        ],
        col_widths=[3.5, 5.5, 7.0],
    )

    add_para(doc, "Camera retry logic", style="Heading 3")
    add_bullets(doc, [
        "NotAllowedError → \"Camera access denied. Please grant permission and reload.\" (no retry)",
        "NotFoundError → \"No camera found.\" (no retry)",
        "NotReadableError or OverconstrainedError → up to 3 retries with 2 s delay between attempts",
    ])


def section_mlops(doc: Document) -> None:
    add_para(doc, "10. MLOps and Training Pipeline", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "All training is reproducible from training/config.yaml, the single source of truth consumed "
        "by both the Jupyter notebook and the standalone scripts. ClearML provides experiment "
        "tracking, dataset versioning, an agent-based remote-execution model, and the HPO "
        "infrastructure used for hyper-parameter optimisation.",
    )

    add_para(doc, "Pipeline steps", style="Heading 3")
    add_table(
        doc,
        ["Step", "Queue", "Function"],
        [
            ["1. Data Prep", "cpu", "Fetch dataset by ID, validate data.yaml, compute class weights"],
            ["2. Train", "gpu", "Ultralytics YOLO training (150 epochs, AdamW, cosine LR, early stop on val mAP)"],
            ["3. Evaluate", "cpu", "Validation metrics + quality gate (mAP@50 ≥ 0.40)"],
            ["4. Export", "cpu", "ONNX export, simplifier, SHA-256 checksum, ClearML artifact upload"],
        ],
        col_widths=[3.5, 2.0, 10.5],
    )

    add_para(doc, "Key training hyperparameters (training/config.yaml)", style="Heading 3")
    add_table(
        doc,
        ["Key", "Value", "Rationale"],
        [
            ["model_weights", "yolo26n.pt", "Pretrained nano backbone, ≈ 2.57 M params"],
            ["epochs", "150", "Nano models converge slowly; complemented by patience"],
            ["patience", "30", "Early-stop guard against overfitting late in training"],
            ["batch", "16", "Fits ~6 GB VRAM; HPO may try 8 or 32"],
            ["imgsz", "640", "Must match ONNX Runtime Web input size in the browser"],
            ["optimizer", "AdamW", "Better convergence than SGD for nano models in our runs"],
            ["lr0 / lrf", "0.001 / 0.01", "Final LR = lr0 × lrf = 1e-5 with cosine annealing"],
            ["warmup_epochs", "5", "Linear warmup prevents early divergence"],
            ["mosaic / mixup", "1.0 / 0.0", "Mosaic on; mixup off (limited gains for nano)"],
            ["export_format", "onnx", "WASM-compatible; not tfjs"],
            ["export_opset", "17", "Maximum opset with full WASM operator coverage"],
            ["export_simplify", "true", "Smaller, faster ONNX file"],
            ["min_map50", "0.40", "Quality gate enforced in evaluation step"],
        ],
        col_widths=[3.5, 3.0, 9.5],
    )

    add_para(doc, "Hyperparameter optimisation", style="Heading 3")
    add_para(
        doc,
        "training/hpo.py launches an Optuna TPE optimiser via ClearML's HyperParameterOptimizer. "
        "The search space spans lr0, lrf, optimizer, batch, weight_decay, warmup_epochs, mosaic, "
        "scale, mixup, and cos_lr. The default budget is 20 trials over an 8-hour wall-clock cap "
        "with up to 2 concurrent tasks. The best parameters are written back to config.yaml so the "
        "next full training run inherits them automatically.",
    )

    add_para(doc, "Remote agents and queues", style="Heading 3")
    add_bullets(doc, [
        "queue_gpu = \"gpu\" — used by training and HPO trials.",
        "queue_cpu = \"cpu\" — used by data prep, evaluation, and export.",
        "Start an agent with: clearml-agent daemon --queue gpu cpu --gpus 0",
        "Run the pipeline remotely with: python training/pipeline.py --remote",
    ])


def section_cicd(doc: Document) -> None:
    add_para(doc, "11. Continuous Integration and Deployment", style="Heading 1")
    add_section_divider(doc)

    add_para(doc, "GitHub Actions workflows", style="Heading 3")
    add_table(
        doc,
        ["Workflow", "Trigger", "Purpose"],
        [
            ["frontend-ci.yml", "PR to main affecting src/ or public/", "npm run lint and npm run build"],
            ["training-config-validate.yml", "PR to main affecting training/config.yaml", "Schema and range validation"],
            ["model-download.yml", "Manual dispatch", "Download ONNX from ClearML and open a PR with the new artifact"],
            ["pages-deploy.yml", "Push to main", "Vite build and Cloudflare Pages deploy"],
        ],
        col_widths=[4.0, 5.0, 7.0],
    )

    add_para(doc, "Deployment topology", style="Heading 3")
    add_para(
        doc,
        "VoiceEye is a static SPA served from Cloudflare Pages with a single Pages Function under "
        "/api/vlm. The static bundle includes the 12 MB ONNX Runtime WASM and the ≈ 9 MB YOLO "
        "model file, both well below the 25 MB per-file limit. public/_headers ensures correct "
        "WASM Content-Type, year-long immutable caching for hashed assets, SPA-fallback for "
        "unknown paths, and a strict Permissions-Policy locking camera, microphone, "
        "accelerometer, and gyroscope to the site origin.",
    )

    add_para(doc, "Quota protection on /api/vlm", style="Heading 3")
    add_bullets(doc, [
        "Origin allow-list — the function rejects requests whose Origin header is outside VLM_ALLOWED_ORIGINS.",
        "Cloudflare Rate Limiting Rule — recommended setting: 20 requests / 1 min / IP, blocked for 10 min on breach.",
        "Optional Cloudflare Access — invitation-only login if the demo must be private.",
    ])


def section_testing(doc: Document) -> None:
    add_para(doc, "12. Testing Strategy", style="Heading 1")
    add_section_divider(doc)

    add_para(doc, "Unit and integration tests (Vitest)", style="Heading 3")
    add_table(
        doc,
        ["Suite", "Test count", "Focus"],
        [
            ["src/utils/tracker.test.ts", "17", "IoU matching, age decay, eviction order, velocity"],
            ["src/utils/attention.test.ts", "30", "Priority, budget, cluster, cooldown, tier-1 override"],
            ["src/utils/distance.test.ts", "12", "Pinhole-model accuracy, edge cases, clamps"],
            ["src/utils/inferenceMetrics.test.ts", "11", "Circular buffer, p95, counter resets"],
            ["src/hooks/useVoiceRecognition.test.ts", "—", "Wake word variants, command parser"],
        ],
        col_widths=[6.5, 2.5, 7.0],
    )

    add_para(doc, "Manual on-device test plan", style="Heading 3")
    add_numbered(doc, [
        "Run npm install (postinstall copies WASM), drop yolo26n.onnx into public/models/.",
        "Run npm run phone and open the HTTPS URL on a real phone. Grant camera and microphone permissions.",
        "Verify bounding boxes appear with {class} #{id} ~Xm labels, colour-coded by proximity.",
        "Verify each object is announced only once until it leaves and re-enters frame.",
        "Walk toward the camera — verify \"Warning: <class> approaching\" fires when area is growing.",
        "Say \"Voice Eye, describe\" — confirm beep + \"Got it\" + Slow Lane response.",
        "Open Settings — change TTS speed and verbosity; verify they take effect immediately.",
        "Disable Haptics — verify no vibration on close objects.",
        "Run npm run build — confirm TypeScript compiles with zero errors.",
    ])


def section_perf(doc: Document) -> None:
    add_para(doc, "13. Performance, Memory, and Battery", style="Heading 1")
    add_section_divider(doc)
    add_bullets(doc, [
        "Singleton OffscreenCanvas and pre-allocated Float32Array eliminate per-frame allocations in the YOLO preprocessor.",
        "ONNX input and output tensors are disposed in try/finally to prevent WASM heap leaks during long sessions.",
        "Inference is throttled to 10 FPS via performance.now(); the rAF loop keeps rendering at the display refresh.",
        "MAX_RENDERED_BOXES = 8 caps DOM mutation in the overlay; further phases plan to migrate to a single canvas overlay.",
        "Cooldown garbage collection drops state-map entries older than COOLDOWN_GC_MS each frame, bounding memory.",
        "Inference metrics summary every 300 frames lets users (and developers) profile real-device latency without DevTools.",
    ])
    add_callout(
        doc,
        "Adaptive throttling (planned)",
        "Phase 3 introduces per-device latency profiling: if average inference latency exceeds 150 ms, "
        "the loop reduces target FPS; if it stays below 50 ms, it increases. The device profile is "
        "persisted to localStorage so it does not need to be re-measured on every launch.",
        bg=ACCENT_BG_HEX,
        accent=INDIGO,
    )


def section_privacy(doc: Document) -> None:
    add_para(doc, "14. Privacy and Security Posture", style="Heading 1")
    add_section_divider(doc)
    add_bullets(doc, [
        "All Fast Lane inference runs on the device; no frames are uploaded.",
        "Slow Lane uploads a single 384 px JPEG only when the user explicitly invokes a voice or tap command, and only to an endpoint configured by the operator (local Ollama or scoped Workers AI Function).",
        "The Pages Function /api/vlm enforces an origin allow-list to prevent third-party sites from consuming Workers AI quota.",
        "public/_headers sets a strict Permissions-Policy that locks the camera, microphone, accelerometer, and gyroscope to the site origin.",
        "Settings (TTS speed, verbosity, sensitivity, haptics) are stored in localStorage; no telemetry, cookies, or analytics are sent off-device by default.",
        "Optional opt-in anonymised telemetry is on the Phase 5 roadmap and is explicitly scoped to never include images or location data.",
    ])


def section_limitations(doc: Document) -> None:
    add_para(doc, "15. Known Limitations", style="Heading 1")
    add_section_divider(doc)
    add_bullets(doc, [
        "Phone FPS, memory, battery, and thermal behaviour still need formally measured benchmark results across reference devices.",
        "Slow Lane response quality varies between local Qwen3-VL and Cloudflare Workers AI; a head-to-head comparison is queued for Sprint 3.",
        "SpeechRecognition API support is uneven across browsers; iOS Safari requires a user gesture to start listening on each session.",
        "The pinhole-camera distance estimator assumes a fixed ≈ 70° vertical FOV and known average object heights — accuracy varies up to ≈ 30 % per device.",
        "YOLO26n is trained on COCO classes only; it does not detect kerbs, steps, puddles, or construction zones, which are exactly the classes most relevant to the target users (planned: ground-hazard fine-tune in Phase 4).",
        "Float32 only — the WASM backend does not support float16, so int8 quantisation is the next compression step (Phase 3).",
        "Python data-prep tests require pytest in a restored environment; CI currently validates config syntactically rather than running the full data-prep test suite.",
    ])


def section_roadmap(doc: Document) -> None:
    add_para(doc, "16. Implementation Plan and Roadmap", style="Heading 1")
    add_section_divider(doc)
    add_para(
        doc,
        "The roadmap is organised into five phases. Phase 1 (Foundation and Quality) is complete. "
        "Phases 2 through 5 prioritise accessibility excellence, performance, and ecosystem growth.",
    )

    add_para(doc, "Phase 1 — Foundation and Quality (complete)", style="Heading 3")
    add_bullets(doc, [
        "Vitest infrastructure and 35+ unit tests for tracker, distance, and inferenceMetrics.",
        "App.tsx decomposed from 446 to 165 lines; logic moved into custom hooks.",
        "React ErrorBoundary with TTS feedback and tap-to-reload.",
        "All magic numbers centralised into src/config.ts (40+ constants).",
        "VLM 15-second timeout + identical-request deduplication.",
        "TypeScript strict null checks enabled.",
        "Spoken error feedback for all three VLM failure paths.",
    ])

    add_para(doc, "Phase 2 — User Experience (Weeks 4–6)", style="Heading 3")
    add_table(
        doc,
        ["Task", "Priority", "Effort"],
        [
            ["Periodic \"thinking\" cue during VLM processing", "High", "0.5 day"],
            ["\"Voice Eye, help\" command", "High", "1 day"],
            ["\"What's around me\" summary command", "High", "1 day"],
            ["Offline fallback + service-worker model precache", "High", "2 days"],
            ["High-contrast theme toggle", "Medium", "2 days"],
            ["Confidence level in announcements", "Medium", "1 day"],
            ["Front/back camera flip toggle", "Medium", "1 day"],
            ["Persistent error indicators (system health)", "Medium", "1 day"],
            ["VLM progress feedback at 0/5/10 s", "Medium", "1 day"],
        ],
        col_widths=[10.0, 3.0, 3.0],
    )

    add_para(doc, "Phase 3 — Performance and Robustness (Weeks 7–9)", style="Heading 3")
    add_table(
        doc,
        ["Task", "Priority", "Effort"],
        [
            ["Single canvas overlay for bounding boxes", "Medium", "2 days"],
            ["Adaptive inference throttling", "High", "2 days"],
            ["ONNX model quantisation (float32 → int8)", "High", "2 days"],
            ["Battery-aware mode (Battery Status API)", "Medium", "1 day"],
            ["Inference metrics export (JSON)", "Low", "1 day"],
            ["WASM cold-start optimisation + load progress", "Medium", "1 day"],
            ["getImageData buffer reuse", "Low", "1 day"],
        ],
        col_widths=[10.0, 3.0, 3.0],
    )

    add_para(doc, "Phase 4 — Accessibility Excellence (Weeks 10–12)", style="Heading 3")
    add_table(
        doc,
        ["Task", "Priority", "Effort"],
        [
            ["i18n + Spanish, Mandarin, Arabic", "High", "3 days"],
            ["VoiceOver / TalkBack compatibility audit", "Critical", "2 days"],
            ["Per-device FOV calibration (credit-card method)", "Medium", "2 days"],
            ["Social VLM mode (\"Voice Eye, who's there?\")", "Medium", "2 days"],
            ["Ground-hazard detection (curbs, steps, puddles)", "High", "3 days"],
            ["Customisable haptic patterns", "Low", "1 day"],
            ["Speech-rate auto-adjustment by urgency", "Medium", "1 day"],
        ],
        col_widths=[10.0, 3.0, 3.0],
    )

    add_para(doc, "Phase 5 — Ecosystem and Scale (Weeks 13+)", style="Heading 3")
    add_table(
        doc,
        ["Task", "Priority", "Effort"],
        [
            ["Recruit 5–10 visually impaired beta testers", "Critical", "Ongoing"],
            ["Opt-in anonymised telemetry", "High", "2 days"],
            ["Automated model-retraining pipeline", "Medium", "3 days"],
            ["Docker container for training", "Medium", "2 days"],
            ["Depth estimation (MiDaS / DepthAnything)", "High", "3 days"],
            ["WebGPU backend for ONNX Runtime", "Medium", "2 days"],
            ["Smart-cane Bluetooth integration (research)", "Low", "Research"],
            ["Indoor navigation + landmark detection (research)", "Low", "Research"],
        ],
        col_widths=[10.0, 3.0, 3.0],
    )


def section_success(doc: Document) -> None:
    add_para(doc, "17. Success Criteria", style="Heading 1")
    add_section_divider(doc)
    add_table(
        doc,
        ["Metric", "Current", "Phase 2 target", "Phase 4 target"],
        [
            ["Test coverage", "≈ 80 % on utils", "80 % utils, 50 % hooks", "80 %+ overall"],
            ["TypeScript strict", "Enabled", "Maintained", "Maintained"],
            ["Lighthouse PWA score", "≈ 60", "90+", "95+"],
            ["Languages supported", "1 (English)", "1", "4"],
            ["WCAG compliance", "Partial AA", "AA", "AAA"],
            ["User studies", "0", "0", "5–10 participants"],
            ["Inference FPS (mid-range phone)", "≈ 8–10", "≈ 8–10", "12–15 (WebGPU)"],
            ["VLM timeout handling", "15 s", "15 s", "15 s"],
            ["Offline Fast Lane", "No", "Yes", "Yes"],
        ],
        col_widths=[5.0, 3.5, 3.5, 3.5],
    )


def section_appendix_voice(doc: Document) -> None:
    add_para(doc, "18. Appendix A — Voice Commands", style="Heading 1")
    add_section_divider(doc)
    add_table(
        doc,
        ["Command", "Mode", "Action"],
        [
            ["\"Voice Eye, describe\"", "Slow Lane / Describe", "VLM scene description spoken aloud"],
            ["\"Voice Eye, read\"", "Slow Lane / OCR", "Reads visible text, documents, signs, or labels"],
            ["\"Voice Eye, find <object>\"", "Slow Lane / Find", "Searches the current frame for the named object"],
            ["Wake word only", "Acknowledgement", "880 Hz beep + \"Got it\" while the parser awaits the rest"],
        ],
        col_widths=[5.5, 4.0, 6.5],
    )


def section_appendix_settings(doc: Document) -> None:
    add_para(doc, "19. Appendix B — Settings Reference", style="Heading 1")
    add_section_divider(doc)
    add_table(
        doc,
        ["Setting", "Range / values", "Default", "Wired to"],
        [
            ["Voice speed (TTS rate)", "0.5 – 2.0", "1.0", "All speak() calls"],
            ["Detection sensitivity", "0.3 – 0.8", "0.5", "confThreshold in runYolo()"],
            ["Verbosity", "Quiet / Normal / Detailed", "Normal", "Attention budget K (1 / 3 / 6)"],
            ["Haptic feedback", "On / Off", "On", "All navigator.vibrate() calls"],
        ],
        col_widths=[4.0, 5.0, 3.0, 4.0],
    )


def section_appendix_files(doc: Document) -> None:
    add_para(doc, "20. Appendix C — File Inventory", style="Heading 1")
    add_section_divider(doc)
    add_para(doc, "Top-level repository layout", style="Heading 3")
    add_table(
        doc,
        ["Path", "Purpose"],
        [
            ["src/", "React + TypeScript application"],
            ["public/", "Static assets, PWA icons, ONNX model, manifest"],
            ["functions/api/vlm.ts", "Cloudflare Pages Function — Workers AI VLM proxy"],
            ["training/", "ClearML training pipeline (Python)"],
            ["mlops_clearml_yolo.ipynb", "Interactive training notebook"],
            ["docs/", "Architecture diagrams and demo script"],
            ["scripts/", "DOCX/PPTX deck generators (python-docx / python-pptx)"],
            [".github/workflows/", "CI/CD pipelines"],
            ["wrangler.toml", "Cloudflare Pages binding configuration"],
            ["vite.config.ts", "Build, dev proxy, PWA configuration"],
            ["package.json", "Frontend scripts and dependencies"],
            ["start.sh / start.bat", "One-click local launcher"],
            ["README.md", "Public project documentation"],
            ["IMPLEMENTATION_PLAN.md", "Phase 1–5 roadmap"],
            ["CLAUDE.md", "AI-assistant guide to the codebase"],
        ],
        col_widths=[5.5, 10.5],
    )


def section_glossary(doc: Document) -> None:
    add_para(doc, "21. Appendix D — Glossary", style="Heading 1")
    add_section_divider(doc)
    add_table(
        doc,
        ["Term", "Meaning"],
        [
            ["PWA", "Progressive Web App — installable, offline-capable web application"],
            ["YOLO26n", "Nano variant of the YOLO26 detection family (~2.57 M params)"],
            ["ONNX Runtime Web", "Browser runtime executing ONNX models in WebAssembly or WebGPU"],
            ["VLM", "Vision-Language Model — accepts an image and a text prompt, returns text"],
            ["Ollama", "Local LLM/VLM server with a REST API on port 11434"],
            ["Workers AI", "Cloudflare's serverless inference platform on the edge"],
            ["Fast Lane", "The continuous detection + tracking + alert path"],
            ["Slow Lane", "The on-demand VLM path (describe / read / find)"],
            ["IoU", "Intersection-over-Union — bounding-box overlap metric used by the tracker"],
            ["EMA", "Exponential moving average — used to smooth velocity estimates"],
            ["TTS / SR", "Text-to-Speech / Speech Recognition (Web Speech API)"],
            ["Quality gate", "An automated threshold (mAP@50 ≥ 0.40) that blocks model promotion"],
            ["HPO", "Hyper-parameter optimisation (Optuna TPE via ClearML)"],
        ],
        col_widths=[3.5, 12.5],
    )


# ── Build orchestration ───────────────────────────────────────────────
def build() -> None:
    doc = Document()
    set_page_margins(doc)
    configure_styles(doc)

    add_cover(doc)
    add_toc(doc)

    section_executive_summary(doc)
    section_problem(doc)
    section_architecture(doc)
    section_tech_stack(doc)
    section_fast_lane(doc)
    section_slow_lane(doc)
    section_voice(doc)
    section_app_structure(doc)
    section_resilience(doc)
    section_mlops(doc)
    section_cicd(doc)
    section_testing(doc)
    section_perf(doc)
    section_privacy(doc)
    section_limitations(doc)
    section_roadmap(doc)
    section_success(doc)
    section_appendix_voice(doc)
    section_appendix_settings(doc)
    section_appendix_files(doc)
    section_glossary(doc)

    doc.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    build()
