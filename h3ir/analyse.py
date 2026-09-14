"""Stage B: AssetCards. The expensive, reusable, cached part.

This is where the hosted stage's apparent magic actually lives. Its verbosity tracks how
much reference material there is to inventory: MiniMax's published T2VA description is 249
words with no references, their Ref2VA edit is 238, but their I2VA is 533 words for a single
STATIC shot -- because that one had an image to catalogue. That is an asset inventory
rendered into prose, and it is exactly reproducible locally.

Cached on content hash + analyzer version + model, so swapping one reference re-analyses one
asset and re-uses everything else.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .backend import Backend, user_message
from .config import get_config
from .models import AssetCard, AssetKind, AssetRef, Role

log = logging.getLogger("h3ir.analyse")

# 2: subjects split into `attributes` (identity) and `pose` (transient). Bumping this is not
# optional when the card's CONTRACT changes -- version 1 cards fold pose into attributes, and
# reusing one silently reintroduces the contamination the split exists to prevent. That is how
# this bump was found: R12 kept firing on a regenerated artifact because the card was cached.
# 3: video cards are built from SAMPLED FRAMES. Version 2 video cards were produced by handing the
# model the .mp4 itself in an `image_url` field -- so it received a base64 video blob where an image
# belongs and returned a card describing nothing in particular. Those cards must not be reused, which
# is the whole reason this constant exists.
# 4: `characterisation` split out of an audio card's `summary`. Version 3 audio cards fold the role
# prefix and the duration into the summary, and the draft appended that whole string to the
# soundscape -- so reusing one puts asset provenance back into a content section.
ANALYZER_VERSION = "5"

# Enough to see a change without paying for a filmstrip. Sampled at 10/50/90% rather than at the ends
# because the first and last frames of a real clip are routinely black or mid-fade.
VIDEO_FRAME_FRACTIONS = (0.1, 0.5, 0.9)

IMAGE_SCHEMA = {
    "title": "ImageCard",
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "subjects", "environment", "lighting", "framing", "style",
                 "visible_text", "composition", "is_reference_sheet"],
    "properties": {
        "summary": {"type": "string"},
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "descriptor", "attributes", "pose"],
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["person", "animal", "object", "environment", "style"]},
                    "descriptor": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "pose": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "environment": {"type": "string"},
        "lighting": {"type": "string"},
        "palette": {"type": "array", "items": {"type": "string"}},
        "framing": {"type": "string"},
        "style": {"type": "string"},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "composition": {"type": "string", "enum": ["bare_plate", "composed_scene", "unknown"]},
        "is_reference_sheet": {"type": "boolean"},
    },
}

VIDEO_SCHEMA = {
    **IMAGE_SCHEMA,
    "title": "VideoCard",
    "required": IMAGE_SCHEMA["required"] + ["camera"],
    "properties": {**IMAGE_SCHEMA["properties"], "camera": {"type": "string"}},
}

# The one question only a video can be asked, and the reason it is asked in plain words rather
# than against `models.CAMERA_TYPES`. Three frames at 10/50/90% can show that the framing tightens;
# they cannot separate a Push In from a Zoom In, or a Pan Left from a Truck Left, because the cue
# that separates each pair is parallax across the frames in between. Offering the closed enum here
# would get a confident member of it every time, and a wrong camera type asserted about a clip the
# writer is told to preserve is worse than no camera type at all.
#
# The measured cost of having neither: on an edit whose source pushes in slowly, the writer wrote
# "static camera" in 3 of 4 seeds. It had been told to keep the clip's camera and had never been
# told what the clip's camera does, and `static` is what a description defaults to.
CAMERA_QUESTION = """

`camera` — what the CAMERA does across these frames, in one short phrase, or exactly "unknown".

Read it from how the framing changes between the frames, not from what the subject does. If the
subject fills more of the frame and less of the room is visible, the camera moved closer; if the
whole scene slides sideways or up, the camera panned or tilted; if the framing is identical in
every frame, it is holding still.

Write what you can see and no more: "the framing tightens on the subject", "the frame slides to
the right", "the framing is unchanged across all three frames". Do NOT name a technique you cannot
tell apart from three stills — closer is closer, whether it was a dolly or a zoom. If the frames
do not settle it, write exactly "unknown". That is a useful answer here and a guess is not.
"""

IMAGE_INSTRUCTIONS = """You catalogue a reference image for a video-generation pipeline.

Record only what is actually visible. Never guess a name, a brand, a place or a backstory.

For each distinct person, animal or object that could be reused in a video, give a descriptor
("the young man", "the black lamb") and then split what you see into TWO separate lists. The
split matters more than the detail:

`attributes` — IDENTITY. What is true of this subject in any photograph of them: hair, beard,
build, age, garment by type and colour, accessories, markings, material, wear. These are the
things a generator needs in order to redraw them recognisably. Aim for five to eight.

`pose` — TRANSIENT. What is true only of THIS photograph: stance, what they are doing with their
hands or limbs, gesture, facial expression, gaze direction, where they sit in the frame, the
camera angle. "Arms crossed", "hands in pockets", "seen from a low angle" belong here and NEVER
in attributes.

OBSERVE, NEVER INTERPRET. Write only what is literally visible and never what it means. "Hands
loosely closed at his sides" is an observation. "Fists clenched in a fighting stance" is a
narrative inference about intent, and it is wrong: it turned a walking posture into combat and
filed it under identity, so the character then arrived in a corridor braced to fight. Never name
a stance ("fighting stance", "heroic pose", "power stance"), never name an emotion or intent
("determined", "menacing", "confident", "ready to strike"), never say what someone is about to
do. If you cannot see it, it is not there.

REFERENCE SHEETS. If the image is a character sheet, turnaround, model sheet or contact sheet --
one subject repeated across several panels, usually with labels and a plain studio backdrop --
then set `is_reference_sheet` true and describe **ONE** subject. Eight views of one man is one
man. The grid, the panel borders, any text burned into the image (FRONT, LEFT PROFILE, BACK) and
the studio backdrop are properties of the SHEET, not of the character: never put them in
`attributes`, `environment`, `visible_text` or anywhere else. Use the multiple views to describe
the character more completely -- hair from the back, the shoe from the side -- which is what the
sheet is for.

`composition`: "bare_plate" if the subject sits alone against a plain or empty background;
"composed_scene" if it is already staged as a finished frame with a full setting.

`visible_text`: every legible string, exactly as written, in its original language.

Describe the environment, the lighting, the framing and the visual style separately from the
subjects. Return JSON only."""

# There is no audio schema and no audio model call, deliberately. See analyse_audio.
def _cache_path(key: str) -> Path:
    d = get_config().paths.cache_dir() / "cards"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _cache_key(ref: AssetRef, model: str) -> str:
    return hashlib.sha256(
        f"{ref.sha256}|{ANALYZER_VERSION}|{model}|{ref.kind.value}".encode()).hexdigest()[:24]


def load_cached(ref: AssetRef, model: str) -> AssetCard | None:
    p = _cache_path(_cache_key(ref, model))
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["kind"] = AssetKind(raw["kind"])
        return AssetCard(**raw)
    except Exception:  # noqa: BLE001 - a bad cache entry is not an error, just a miss
        return None


def save_cached(ref: AssetRef, card: AssetCard, model: str) -> None:
    from dataclasses import asdict
    d = asdict(card)
    d["kind"] = card.kind.value
    _cache_path(_cache_key(ref, model)).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def analyse_image(backend: Backend, ref: AssetRef, *, seed: int | None = None) -> AssetCard:
    hint = ""
    if ref.provenance:
        # A generated asset knows how it was made. That is a PRIOR, never a substitute: the
        # model conditions on pixels, and a generated image routinely misses its own prompt.
        src = ref.provenance.get("source_prompt") or ref.provenance.get("edit_instruction")
        if src:
            hint = ("\n\nThis image was machine-generated from the following instruction. Treat it "
                    "as a hint only and correct it against what you actually see:\n" + str(src))
    if ref.note:
        hint += f"\n\nThe caller says this reference is: {ref.note}"

    obj = backend.json_call(
        [{"role": "system", "content": IMAGE_INSTRUCTIONS},
         user_message("Catalogue this image." + hint, [ref.path] if ref.path else None)],
        IMAGE_SCHEMA, required=("summary", "subjects"), seed=seed, max_tokens=6000,
        # The same deep budget as analyse_video, for the same reason: vision is this endpoint's
        # flakiest structured caller, and a card that cannot be built kills the whole brief.
        retries=5)

    return AssetCard(
        sha256=ref.sha256, kind=AssetKind.IMAGE,
        summary=obj.get("summary", ""),
        subjects=obj.get("subjects") or [],
        environment=("" if obj.get("is_reference_sheet") else obj.get("environment", "")),
        lighting=obj.get("lighting", ""),
        palette=obj.get("palette") or [],
        framing=obj.get("framing", ""),
        style=obj.get("style", ""),
        visible_text=([] if obj.get("is_reference_sheet") else (obj.get("visible_text") or [])),
        composition=(ref.composition if ref.composition != "unknown"
                     else obj.get("composition", "unknown")),
        is_reference_sheet=bool(obj.get("is_reference_sheet")),
        analyzer_version=ANALYZER_VERSION,
        model_id=backend.cfg.model,
    )


def analyse_audio(ref: AssetRef, transcript: str = "", *,
                  duration_s: float | None = None) -> AssetCard:
    """Typed metadata only. NO model call -- the model is deaf and must never be asked.

    Three reasons this is a rule and not a preference:

      * The endpoint has a vision tower and no audio tower. Asked about a waveform it cannot
        hear, it does not say so; it invents a plausible description. A confident wrong timbre
        is worse than a missing one, because it reaches the IR as an assertion.
      * The prior-art sweep found the same wall independently ("Qwen3-VL does not inspect saved
        audio waveforms"), and a note that a deaf model gets confused rather than abstaining.
      * H3's own tokenizer emits `"<Audio j>: "` and nothing else -- the audio content never
        enters the conditioning encoder at all. So the IR text is the ONLY channel by which the
        encoder learns what that audio is, which makes an invented description actively harmful
        rather than merely useless.

    What we legitimately know comes from the wiring: the role the caller assigned, the caller's
    note, the duration, and a transcript if one was produced by an actual speech recogniser.
    The prose stage then writes around those facts instead of about the sound.
    """
    role_summary = {
        Role.VOICE_TIMBRE: "a spoken vocal reference supplying voice timbre and delivery",
        Role.BGM: "a background music track",
        Role.MUSIC_STYLE: "a music-style reference supplying instrumentation and tempo",
        Role.BEAT_REFERENCE: "a rhythmic reference supplying beat and tempo",
        Role.SFX: "a sound-effect reference",
    }.get(ref.role, "an audio reference")
    summary = role_summary
    if ref.note:
        summary += f", described by the caller as: {ref.note.strip()}"
    if duration_s or ref.seconds:
        summary += f" ({(duration_s or ref.seconds):.2f}s)"

    language = ""
    if transcript:
        # Only what a recogniser actually reported; never guessed from the filename.
        language = (ref.provenance or {}).get("language", "") if ref.provenance else ""

    return AssetCard(
        sha256=ref.sha256, kind=AssetKind.AUDIO,
        summary=summary + ".",
        # timbre and music stay EMPTY unless something that can hear fills them in. A TRANSCRIPT
        # does not fill them: it gives the words, not the delivery, the tempo, or whether the thing
        # is music at all. So a transcript closes the dialogue half of audio and leaves the sonic
        # half exactly where it was -- only the caller can say "his voice, calm and low".
        timbre="", music="",
        characterisation=(ref.note or "").strip(),
        transcript=transcript, language=language,
        analyzer_version=ANALYZER_VERSION, model_id="none (typed metadata only)")


# --------------------------------------------------------------------------- what a file IS

def sniff_container(path: str | Path) -> str | None:
    """"image", "video", "audio", or None when the bytes are not one this recognises.

    POSITIVE identification only, and that asymmetry is the whole design. A file this cannot place
    gets None and goes through untouched, because refusing a format the vision model might handle
    fine would trade a bad error message for lost work. What it is used for is the one case that is
    decidable: the caller declared `kind: image` and the bytes are an MP4 container, which reached
    the vision endpoint and came back as `HTTP 400: Failed to load image: cannot identify image
    file <_io.BytesIO object>` -- the inference server's internals, forwarded to the caller as a 502,
    for what is a wrong field in their request.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
    except OSError:
        return None
    if head[:8] == b"\x89PNG\r\n\x1a\n" or head[:2] == b"\xff\xd8" or head[:6] in (
            b"GIF87a", b"GIF89a") or head[:2] == b"BM":
        return "image"
    if head[:4] == b"RIFF":
        # WEBP, WAVE and AVI all open with RIFF; the type is the four bytes after the size.
        return {b"WEBP": "image", b"WAVE": "audio", b"AVI ": "video"}.get(head[8:12])
    if head[4:8] == b"ftyp":
        # ISO base media: mp4, m4v, mov, and also m4a, which is audio in the same container. The
        # brand cannot be trusted to separate them (an m4a can carry `isom`), so this reports
        # "video" and callers use it only where that ambiguity does not matter.
        return "video"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "video"          # matroska / webm
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio"
    if head[:4] in (b"fLaC", b"OggS"):
        return "audio"
    return None


def image_mime(path: str | Path) -> str | None:
    """The image type these BYTES are, or None when they are not an image this recognises.

    Separate from `sniff_container`, which answers the coarser question of which analyser a file
    belongs to. This one exists because a data URL has to declare a type and the name is not
    evidence: an uploaded attachment is stored under its content hash and has no extension at all,
    so `mimetypes.guess_type` on it returns nothing and the fallback declared every one of them
    `image/png`. A JPEG announced as a PNG is the kind of thing an inference server either forgives
    or reports in its own internal terms, and which one is not this layer's decision to bet on.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:2] == b"\xff\xd8":
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:2] == b"BM":
        return "image/bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_size(path: str | Path) -> tuple[int, int] | None:
    """Pixel dimensions from the file header. None for anything not recognised.

    `AssetRef.px` was populated in exactly one place in the codebase -- the eval suite -- so the
    service, the CLI and the node all left it None. Everything downstream that reads it therefore
    took its fallback branch: `plan.rows_for_image` reported the canvas figure for every reference
    regardless of its real size, so `sizing: "max"` and `sizing: "match"` published the SAME row
    cost for the same plate (1008 for a 1400x933 image, where max sizing is really 1276 and a
    4000x3000 plate would be 5440), and `mode.infer_mode`'s aspect-mismatch downgrade could never
    fire because it reads `a.px`.

    Header parsing rather than a dependency: this package's runtime deps are fastapi, uvicorn,
    pydantic, httpx and tiktoken, and one number per image does not earn Pillow. Formats a
    generative pipeline actually produces: PNG, JPEG, WebP (all three chunk types), GIF, BMP.
    """
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w = int.from_bytes(head[16:20], "big")
                h = int.from_bytes(head[20:24], "big")
                return (w, h) if w and h else None
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return (int.from_bytes(head[6:8], "little"),
                        int.from_bytes(head[8:10], "little"))
            if head[:2] == b"BM":
                fh.seek(18)
                b = fh.read(8)
                return (abs(int.from_bytes(b[0:4], "little", signed=True)),
                        abs(int.from_bytes(b[4:8], "little", signed=True)))
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _webp_size(fh, head)
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(fh)
    except OSError:
        return None
    return None


def _webp_size(fh, head: bytes) -> tuple[int, int] | None:
    chunk = head[12:16]
    if chunk == b"VP8X":
        fh.seek(24)
        b = fh.read(6)
        return (int.from_bytes(b[0:3], "little") + 1, int.from_bytes(b[3:6], "little") + 1)
    if chunk == b"VP8 ":
        fh.seek(26)
        b = fh.read(4)
        return (int.from_bytes(b[0:2], "little") & 0x3FFF,
                int.from_bytes(b[2:4], "little") & 0x3FFF)
    if chunk == b"VP8L":
        fh.seek(21)
        bits = int.from_bytes(fh.read(4), "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


def _jpeg_size(fh) -> tuple[int, int] | None:
    """Walk the segment markers to the first SOF. The dimensions are not at a fixed offset in a
    JPEG, so there is no shortcut that is correct for progressive and EXIF-carrying files."""
    fh.seek(2)
    while True:
        b = fh.read(1)
        while b and b != b"\xff":
            b = fh.read(1)
        marker = fh.read(1)
        while marker == b"\xff":
            marker = fh.read(1)
        if not marker:
            return None
        m = marker[0]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            continue
        size = fh.read(2)
        if len(size) < 2:
            return None
        length = int.from_bytes(size, "big")
        # SOF0..SOF15, excluding DHT (C4), JPG (C8) and DAC (CC), which are not frame headers.
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            body = fh.read(5)
            if len(body) < 5:
                return None
            return (int.from_bytes(body[3:5], "big"), int.from_bytes(body[1:3], "big"))
        fh.seek(length - 2, 1)


class AssetAnalysisError(RuntimeError):
    """An asset could not be analysed. Raised rather than returning an empty card: a card that
    describes nothing is indistinguishable from a card describing something dull, and the compiler
    would build a brief on it without noticing.

    The caller can act on this one: the file is unreadable, or it is not the kind it was attached
    as. `service.create_brief` returns it as 422 with the message intact.
    """


class ToolMissing(AssetAnalysisError):
    """ffmpeg or ffprobe is absent from the machine running the service.

    A separate class rather than a phrase in the message, because the two need different answers
    and the difference is not the caller's to fix: an unreadable file is a 422 they can correct,
    and a missing binary is a 503 about this deployment. Sniffing "not installed" out of the text
    would make the HTTP status depend on the wording of a sentence written for a human.
    """


def _run_ff(argv: list[str], *, timeout: int = 30):
    """Run an ffmpeg-family tool, and say so plainly when it is not installed.

    `subprocess.run` raises `FileNotFoundError` when the binary is absent, which reaches the caller
    as a bare OS error naming a path they never typed. A user who attaches a video without ffmpeg
    installed deserves to be told that, not to debug a traceback -- and the same gap turned a CI
    runner without ffmpeg into a confusing test failure rather than a clear one.
    """
    import subprocess
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise ToolMissing(
            f"{argv[0]} is not installed, and video references need it. Install ffmpeg "
            f"(it provides both ffmpeg and ffprobe) and try again.") from e


def probe_seconds(path: str | Path) -> float:
    """Duration from the file itself, which is ground truth -- a caller's `seconds` is a claim."""
    out = _run_ff(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    if out.returncode != 0 or not out.stdout.strip():
        raise AssetAnalysisError(f"ffprobe could not read {path}: {out.stderr.strip()[:200]}")
    try:
        return float(out.stdout.strip())
    except ValueError as e:
        raise AssetAnalysisError(f"ffprobe returned no duration for {path}") from e


def sample_frames(path: str | Path, sha256: str,
                  fractions: tuple[float, ...] = VIDEO_FRAME_FRACTIONS) -> list[str]:
    """Extract representative frames, cached beside the cards on the asset's content hash.

    Nothing produced frames before this: `analyse_video` accepted a frame list and every caller
    passed none, so it fell through to handing the model the video file as an image.
    """

    p = Path(path)
    if not p.exists():
        raise AssetAnalysisError(f"video reference does not exist: {p}")
    # Keyed on the FRACTIONS as well as the content, for the same reason cards are keyed on
    # ANALYZER_VERSION: change what the artifact means and a cached one must not be reused. Without
    # this, editing VIDEO_FRAME_FRACTIONS silently reuses frames taken at the old timestamps -- which
    # is exactly how a falsification run passed when it should have failed, the cache quietly serving
    # correct frames to a sampler that had been broken on purpose.
    key = hashlib.sha256(repr(tuple(fractions)).encode()).hexdigest()[:8]
    out_dir = get_config().paths.cache_dir() / "frames" / f"{sha256[:16]}-{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = probe_seconds(p)
    made: list[str] = []
    for i, frac in enumerate(fractions):
        dest = out_dir / f"f{i}.jpg"
        if not dest.exists():
            at = max(0.0, min(seconds * frac, max(0.0, seconds - 0.05)))
            r = _run_ff(
                ["ffmpeg", "-nostdin", "-y", "-ss", f"{at:.3f}", "-i", str(p),
                 "-frames:v", "1", "-q:v", "3", str(dest)], timeout=120)
            if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
                log.warning("frame at %.2fs failed for %s: %s", at, p.name,
                            r.stderr.strip()[-200:])
                continue
        made.append(str(dest))
    if not made:
        raise AssetAnalysisError(
            f"could not sample a single frame from {p.name}. A video card built without frames "
            "describes nothing, so this raises instead of returning one.")
    return made


def analyse_video(backend: Backend, ref: AssetRef, frames: list[str] | None = None,
                  *, seed: int | None = None) -> AssetCard:
    """A video card is an image card for representative frames plus its motion.

    Frames are REQUIRED. Passing the video file to a vision model as an image is what the previous
    version did, and it produced a plausible-looking card from a base64 blob the model could not
    read -- silent, because nothing errored and the card had the right shape.
    """
    frames = frames or (sample_frames(ref.path, ref.sha256) if ref.path else [])
    if not frames:
        raise AssetAnalysisError(
            f"no frames for video {ref.sha256[:12]} and no path to sample from")
    obj = backend.json_call(
        [{"role": "system", "content": IMAGE_INSTRUCTIONS +
          f"\n\nThese {len(frames)} frames are sampled in order from across one video clip: the "
          "first is near the start, the last near the end. They are the SAME clip, not separate "
          "references. Record what changes between them -- movement, position, light -- in the "
          "`summary` sentence, and describe the subject once rather than once per frame."
          + CAMERA_QUESTION},
         user_message("Catalogue this clip." + (f"\n\nCaller's note: {ref.note}" if ref.note else ""),
                      frames)],
        VIDEO_SCHEMA, required=("summary", "subjects"), seed=seed, max_tokens=6000,
        # Multi-frame vision is this endpoint's flakiest structured call: measured 2026-08-15, the
        # same clip failed the schema-echo check twice and passed on the third attempt, and one
        # brief burned all three attempts three requests running. Six attempts, because a video
        # card that cannot be built kills the whole brief.
        retries=5)
    return AssetCard(sha256=ref.sha256, kind=AssetKind.VIDEO,
                     summary=obj.get("summary", ""), subjects=obj.get("subjects") or [],
                     environment=obj.get("environment", ""), lighting=obj.get("lighting", ""),
                     palette=obj.get("palette") or [], framing=obj.get("framing", ""),
                     style=obj.get("style", ""), visible_text=obj.get("visible_text") or [],
                     motion=obj.get("summary", ""),
                     camera=_camera_or_blank(obj.get("camera", "")),
                     composition=obj.get("composition", "composed_scene"),
                     frames_seen=len(frames),
                     analyzer_version=ANALYZER_VERSION, model_id=backend.cfg.model)


def _camera_or_blank(said: str) -> str:
    """An unobservable camera is stored as nothing at all, not as the word "unknown".

    A card field that is empty means "not observed" everywhere else in this module, and one that
    carries the literal string "unknown" would be spliced into an ask as a fact about the clip.
    """
    said = (said or "").strip().rstrip(". ")
    return "" if said.lower() in ("", "unknown", "n/a", "none") else said


def _name_the_asset(e: AssetAnalysisError, ref: AssetRef) -> AssetAnalysisError:
    """Say WHICH attachment failed and what it was attached as, keeping the class.

    A brief can carry twelve files. "could not sample a single frame from plate-car.jpg" leaves
    the caller to work out which of them that was and, worse, does not mention the thing that is
    usually wrong: the file is fine and the declared `kind` is not. The most common way to reach
    this is a still attached as `kind: video`, so the hint names that possibility without
    asserting it -- nothing here has established what the file actually is.

    Same class, so `ToolMissing` does not degrade into a plain analysis error and lose its 503.
    """
    where = ref.path or ref.url or f"sha256 {ref.sha256[:12]}"
    msg = f"{str(e).rstrip('. ')} (attached as kind: {ref.kind.value}, {where})"
    if isinstance(e, ToolMissing) or ref.kind is not AssetKind.VIDEO:
        return type(e)(msg + ".")
    return type(e)(msg + ". If it is a still image, attach it with kind: image instead.")


def measure_assets(refs: list[AssetRef]) -> None:
    """Fill in what the file itself says, and refuse a file whose bytes are the wrong kind.

    Two things the layer was guessing at when it did not have to. Dimensions decide the row cost
    this service publishes and the aspect check mode inference makes, and nothing populated them.
    The declared kind decides which analyser runs, and an mp4 declared `image` used to travel all
    the way to the vision endpoint and come back as its internal error, forwarded as a 502.

    In place, because the AssetRef is this layer's own intake object and every stage after intake
    reads `px` off it. Never overwrites a value the caller supplied.
    """
    for ref in refs:
        if not ref.path:
            continue
        sniffed = sniff_container(ref.path)
        if sniffed and sniffed != ref.kind.value and {sniffed, ref.kind.value} <= {"image", "video"}:
            raise AssetAnalysisError(
                f"{Path(ref.path).name} was attached as kind: {ref.kind.value}, and its bytes are "
                f"a {sniffed} file. Attach it with kind: {sniffed}. Sending it to the analyser as "
                f"{ref.kind.value} fails inside the inference server, which reports it in its own "
                "terms rather than yours")
        if ref.kind is AssetKind.IMAGE and ref.px is None:
            ref.px = image_size(ref.path)


def analyse_all(backend: Backend, refs: list[AssetRef], *, use_cache: bool = True,
                seed: int | None = None,
                transcripts: dict[str, str] | None = None) -> dict[str, AssetCard]:
    out: dict[str, AssetCard] = {}
    for ref in refs:
        # An AUDIO card is never cached, in either direction, and that is not a performance
        # trade: `analyse_audio` makes no model call and reads no bytes, so there is nothing to
        # save. Every field in it comes from the REQUEST -- the role picks the summary sentence,
        # the note becomes `characterisation` (the only channel the encoder has for how the audio
        # sounds), the caller's recogniser supplies the transcript -- and none of those is in the
        # cache key, which is `sha256|version|model|kind`. So the first request to attach a wav
        # decided what every later request attaching the same wav would say about it: the live
        # cache entry for one probe's voice reference held `transcript: ''` plus a note from an
        # unrelated request, and that is what reached the writer on seven runs whose caller had
        # supplied the words. Content-addressing is right for an image or a video, where the card
        # IS derived from the bytes; here it addresses content that contributes nothing to the card.
        cacheable = use_cache and ref.kind is not AssetKind.AUDIO
        if cacheable:
            hit = load_cached(ref, backend.cfg.model)
            if hit is not None:
                log.info("card cache hit %s", ref.sha256[:12])
                out[ref.sha256] = hit
                continue
        try:
            if ref.kind is AssetKind.IMAGE:
                card = analyse_image(backend, ref, seed=seed)
            elif ref.kind is AssetKind.AUDIO:
                card = analyse_audio(ref, (transcripts or {}).get(ref.sha256, ""))
            else:
                card = analyse_video(backend, ref, seed=seed)
        except AssetAnalysisError as e:
            raise _name_the_asset(e, ref) from e
        out[ref.sha256] = card
        if cacheable:
            save_cached(ref, card, backend.cfg.model)
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
