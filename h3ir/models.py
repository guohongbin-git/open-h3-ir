"""The contract. Everything that crosses a stage boundary is one of these.

The invariant the whole service rests on: `IRDocument.prompt` is a pure function of the
rest of the document. Re-rendering must reproduce it byte-for-byte (validator rule
X1-render-determinism), so an IR can always be explained by the plan that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .grid import Target, ms_to_timestamp


class Mode(str, Enum):
    T2VA = "t2va"
    I2VA = "i2va"
    FL2VA = "fl2va"
    L2VA = "l2va"
    REF2VA = "ref2va"

    @property
    def checkpoint(self) -> str:
        return "ref2va" if self is Mode.REF2VA else "fl2va"

    @property
    def is_base(self) -> bool:
        return self is not Mode.REF2VA


class AssetKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Role(str, Enum):
    """Why an asset is attached. Task types and retention markers are DERIVED from this,
    so it must be explicit rather than inferred from prose later."""

    FRAME_ANCHOR_FIRST = "frame_anchor_first"
    FRAME_ANCHOR_LAST = "frame_anchor_last"
    SUBJECT = "subject"
    ENVIRONMENT = "environment"
    STYLE = "style"
    STORYBOARD = "storyboard"
    # A video that lends its camera movement and cutting rhythm and nothing else — the
    # structure sibling of STYLE, added when the sentence alone measurably could not keep a
    # clip's contents out of the target video (matrix row 26).
    STRUCTURE = "structure"
    EDIT_SOURCE = "edit_source"
    CONTINUATION_SOURCE = "continuation_source"
    # The two halves of "put this into the clip". Both ride a PICTURE and both require an
    # `edit_source` video in the same brief, because both are statements about what happens to
    # that clip; `check_request` refuses them without one rather than degrading to `subject`.
    #
    # The clip keeps `edit_source`. It already derives `video editing`, the mandated summary
    # opening and the mandated definition line, all of them correct; what was missing was the
    # other half of the sentence -- what the attached picture is FOR. Putting the job on the
    # picture also composes, because two pictures can be placed into one clip and a role on the
    # video could only name one relationship for the whole edit.
    #
    # TWO roles rather than one, by the same test the music pair below was settled on: the
    # retention note and the definition line are derived from the role, and these two cases need
    # different true sentences. A placement ADDS and takes nothing away, so nothing in the clip
    # is transferred anywhere. A replacement REMOVES a figure and hands that figure's position in
    # frame, actions and timing to the new subject -- which is ref-en.txt 4.1's `attribute_transfer`
    # word for word, "referenced characteristics are transferred to a different identifiable
    # target subject", and the only place in this compiler where that marker applies. One role
    # covering both could only write a sentence vague enough to be true of either, and that
    # sentence is what the deterministic draft ships when the writer fails.
    PLACED_SUBJECT = "placed_subject"
    REPLACEMENT_SUBJECT = "replacement_subject"
    VOICE_TIMBRE = "voice_timbre"
    BGM = "bgm"
    # ref-en.txt 2.4 lists five uses for an <Audio N> and three of them had a role: copying the
    # signal (`bgm`), a speaker's timbre and delivery (`voice_timbre`), sound-effect texture
    # (`sfx`). The two with no socket were "Referencing a background-music style" and "Referencing
    # beat, rhythm, or audio continuity", so a caller who wanted either had to attach the track as
    # `bgm`, whose derived bookkeeping says the signal is copied. Measured at five seeds each on the
    # live service: `S6-beat-rhythm` shipped `fully_copy` 5 of 5 and `X9`, whose request says in as
    # many words that nothing from the recording is used, 5 of 5. Both shipped `ready`.
    #
    # TWO roles rather than one, because the retention note and the definition line are derived
    # from the role and the two cases need different true sentences: a style reference says the new
    # score adopts the instrumentation and tempo, and a beat reference says the cutting follows the
    # hits. One role covering both could only write a sentence vague enough to be true of either,
    # and the draft that carries that sentence is what ships when the writer fails.
    #
    # `beat_reference` carries the suffix and `music_style` does not, for a reason rather than by
    # accident: a music style and a voice timbre are properties nobody can mistake for a signal,
    # while "a beat" colloquially IS the track. The suffix is what stops a caller who wants the
    # track PLAYED from picking this instead of `bgm`.
    MUSIC_STYLE = "music_style"
    BEAT_REFERENCE = "beat_reference"
    SFX = "sfx"


VISUAL_MARKERS = ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference")
AUDIO_MARKERS = ("fully_copy", "partially_copy", "reference", "weak_reference")

# The audio roles whose definition IS "a property is referenced, not the signal", against ref-en.txt
# 4.2's `reference` row: "only timbre, rhythm, music style, dialogue content, or sound texture is
# referenced". Contract rather than a local list, because three stages have to agree on it: the
# marker table and the task-type derivation in plan.py, the fact the ask states in prose.py, and the
# rules that refuse a copy claim in validate.py. A role added to one and forgotten in another is how
# a document ends up claiming a copy the wiring does not perform.
AUDIO_REFERENCE_ROLES = (Role.VOICE_TIMBRE, Role.SFX, Role.MUSIC_STYLE, Role.BEAT_REFERENCE)
AUDIO_REFERENCE_ROLE_VALUES = tuple(r.value for r in AUDIO_REFERENCE_ROLES)

# The picture roles whose meaning is a statement about an attached `edit_source` clip. A contract
# rather than a local list for the same reason as the line above: four stages have to agree on it
# -- the intake refusal in compile.py, the marker table and task-type derivation in plan.py, the
# draft's definition and retention lines in render.py, and the every-role guard in the suite. A
# role added to one and forgotten in another is how a document ends up claiming a swap the wiring
# does not perform.
#
# There is no `_VALUES` twin here, unlike the audio pair above. That one exists because a rule
# asks whether EVERY attached audio is reference-only; the swap rules each name one role, and
# `validate.py` compares those the way R28-R30 already do, against the role's own string.
SWAP_ROLES = (Role.PLACED_SUBJECT, Role.REPLACEMENT_SUBJECT)

TASK_TYPES = ("keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference")

# The closed camera vocabulary from base-en.txt §4.3. 12 motion types + amplitude + speed.
CAMERA_TYPES = ("Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
                "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
                "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
                "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
                "Roll Counterclockwise")
AMPLITUDES = ("small", "large")
SPEEDS = ("slow", "fast")

# STABLE_LANGUAGES removed by the source audit: the claim that these eleven are H3's "stably
# supported" languages appears in neither spec document nor in any measurement, and D6 -- the only
# rule that used it -- went with it. If a real list turns up, it needs a citation before it comes back.


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- inputs

@dataclass
class AssetRef:
    """One attached file, before the runtime has assigned it a label."""

    kind: AssetKind
    role: Role
    sha256: str
    path: str | None = None
    url: str | None = None
    note: str | None = None
    px: tuple[int, int] | None = None
    seconds: float | None = None
    frames: int | None = None
    sizing: str = "match"            # match | max
    composition: str = "unknown"     # bare_plate | composed_scene | unknown
    provenance: dict[str, Any] | None = None
    paired_video_sha256: str | None = None   # a soundtrack points at its video
    # Who this picture takes over from, in the caller's own words, and only on a
    # `replacement_subject`. Free text because nothing here can enumerate the people in a clip: the
    # analyser sees three sampled frames of it, so a figure can be absent from all three and walk in
    # later, and any rule resting on a head count would be guessing. The caller is the one who knows.
    replaces: str = ""
    # Did the CALLER name the role, or is it the kind's default? `role` cannot answer that: the
    # service fills an omitted role with `subject` for an image, so an explicit `role: "subject"`
    # and an omitted one arrive identical. Mode inference needs the difference -- a stated
    # `storyboard` is ground truth about what the picture is FOR and outranks a phrase in the
    # request, while an unstated `subject` is only a placeholder and must not block the anchor
    # reading. `AssetIn.role` already promises "Inferred when omitted"; this is what makes that
    # promise keepable.
    role_stated: bool = False


@dataclass
class DialogueLine:
    """User-supplied speech. `text` is pasted verbatim and never passes through a model."""

    text: str
    language: str = "English"
    speaker_hint: str | None = None
    voiceover: bool = False
    speaker_subject: int | None = None   # 1-based <Subject N>; caller-owned binding


@dataclass
class Brief:
    """What a caller sends. One sentence and some files is a complete request."""

    intent: str
    assets: list[AssetRef] = field(default_factory=list)
    seconds: float = 5.0
    aspect: str = "16:9"
    canvas: tuple[int, int] | None = None   # pin the exact canvas; else derived from aspect
    megapixels: float | None = None         # size ask in MP; None = H3's native 768 short edge
    dialogue: list[DialogueLine] = field(default_factory=list)
    onscreen_text: list[str] = field(default_factory=list)
    shots: int | None = None          # caller constraint, not the plan
    loras: list[dict[str, Any]] = field(default_factory=list)
    mode: Mode | None = None          # caller override; normally inferred
    effort: str = "standard"          # fast | standard | max
    constraints: list[str] = field(default_factory=list)
    silent: bool = False
    # restrained | balanced | bold -- how much the writer may add beyond what was asked. An explicit
    # input, never inferred from the request; see creativity.py for why inference was rejected.
    creativity: str = "balanced"
    # Whose taste fills what the request and the references leave open. Both empty is the default
    # and is what every brief written before this existed compiles at. TWO fields rather than a
    # union, because this crosses an HTTP boundary: `director` names one of the profiles that ship
    # and `director_profile` carries a caller's own, which is a name and a paragraph. A field that
    # is sometimes a string and sometimes an object is a schema two callers will disagree about.
    # The ComfyUI node only ever fills the second: a shipped profile is loaded into the box on the
    # canvas and is the user's to edit from that moment. See director.py.
    director: str = ""
    director_profile: dict[str, Any] | None = None

    def hash(self) -> str:
        return _sha(asdict(self))


# --------------------------------------------------------------------------- stage B

@dataclass
class AssetCard:
    """What one asset actually contains. Cached on content hash; the expensive, reusable part."""

    sha256: str
    kind: AssetKind
    summary: str = ""
    subjects: list[dict[str, Any]] = field(default_factory=list)
    environment: str = ""
    lighting: str = ""
    palette: list[str] = field(default_factory=list)
    framing: str = ""
    style: str = ""
    visible_text: list[str] = field(default_factory=list)
    motion: str = ""
    # What the CAMERA does across a video card's sampled frames, in plain words, or empty when the
    # frames do not settle it. Video only, and deliberately not a member of `CAMERA_TYPES`: three
    # frames cannot separate a push from a zoom, and the closed vocabulary is the planner's to
    # choose for a GENERATION. On an edit nobody chooses it -- the clip already has one -- so what
    # is wanted here is an observation, and an unobservable one stays empty.
    camera: str = ""
    # How many frames a video card was built from. 0 on an image or audio card. Recorded because a
    # card built from three frames must not read as one built from the whole clip.
    frames_seen: int = 0
    # The caller's own words about what an asset SOUNDS like, verbatim, with no role prefix and no
    # duration bolted on. Kept separate from `summary` because the summary is provenance about the
    # asset and this is a fact about the video's content -- they belong in different sections, and
    # folding them together is what put "described by the caller as: ... (6.00s)" into
    # overall_soundscape, where the spec wants the target video's ambience.
    characterisation: str = ""
    composition: str = "unknown"
    # A character sheet / turnaround: ONE subject across several panels. Its grid, labels and
    # studio backdrop belong to the sheet and must never reach the brief.
    is_reference_sheet: bool = False
    # audio only
    transcript: str = ""
    language: str = ""
    timbre: str = ""
    music: str = ""
    analyzer_version: str = "1"
    model_id: str = ""

    def hash(self) -> str:
        return _sha(asdict(self))


# --------------------------------------------------------------------------- stage C

@dataclass
class ManifestEntry:
    """An asset with its runtime-assigned label. Slot order IS the label numbering."""

    slot: int
    label: str
    kind: AssetKind
    sha256: str
    wiring: str
    role: Role
    px: tuple[int, int] | None = None
    seconds: float | None = None
    frames: int | None = None
    sizing: str = "match"
    rows: int = 0
    paired_with: str | None = None
    # The caller's own words about the asset. Reaches the renderer this way because
    # render_subject_definitions sees the plan and not the cards, and for audio it is the ONLY thing
    # that can describe the sound -- nothing in this system can hear.
    characterisation: str = ""
    composition: str = "unknown"
    # A `replacement_subject` picture's statement of who it takes over from, in the caller's words.
    # Carried on the manifest because the binding runs off the manifest and the subject list and
    # never sees the brief.
    replaces: str = ""


@dataclass
class SubjectPlan:
    label: str                       # "<Subject 1>"
    kind: str                        # person | environment | object | style | action
    sources: list[str]               # grounded labels it is drawn from
    descriptor: str                  # "the young man"
    # IDENTITY only. What is true of the subject in any photograph of them.
    attributes: list[str] = field(default_factory=list)
    # TRANSIENT: recorded so we know what the plate showed, and asserted in the IR only when the
    # plate IS a frame of the video (a frame anchor). Otherwise the request owns the action.
    pose: list[str] = field(default_factory=list)
    pose_licensed: bool = False
    retention: str = "fully_preserved"
    retention_note: str = ""
    appears_in: list[int] = field(default_factory=list)
    # Set only on the figure a `replacement_subject` picture takes over from: the label of the
    # subject that inherits its position in frame, actions and timing. It is what makes
    # `attribute_transfer` a legal claim on this line -- ref-en.txt 4.1 requires the transfer to
    # land on "a different identifiable target subject", and this names which one. Empty
    # everywhere else, which is every brief that predates the swap roles.
    taken_over_by: str = ""


@dataclass
class SpeakerPlan:
    sid: str                         # "(S1)"
    subject: str | None              # "<Subject 1>" or None
    voice: str = ""
    voice_ref: str | None = None     # "<Audio 2>"
    onscreen: bool = True
    descriptor: str = ""


@dataclass
class CameraMove:
    """Chosen from the closed vocabulary by the planner, rendered canonically. The model
    never writes camera prose, which is what makes the vocabulary A/B a template flag."""

    type: str
    amplitude: str | None = None
    speed: str | None = None

    def phrase(self, style: str = "canonical") -> str:
        if style == "prose":
            return ""
        verb = {
            "Zoom In": "zooms in", "Zoom Out": "zooms out",
            "Push In": "pushes in", "Pull Out": "pulls out",
            "Pan Left": "pans left", "Pan Right": "pans right",
            "Truck Left": "trucks left", "Truck Right": "trucks right",
            "Tilt Up": "tilts up", "Tilt Down": "tilts down",
            "Pedestal Up": "rises on a pedestal move", "Pedestal Down": "lowers on a pedestal move",
            "Arc Shot": "arcs around the subject", "Tracking Shot": "tracks with the subject",
            "Static Shot": "holds a static shot",
            "Shake Slightly": "shakes slightly", "Shake Strongly": "shakes strongly",
            "POV": "takes the subject's point of view",
            "Roll Clockwise": "rolls clockwise", "Roll Counterclockwise": "rolls counterclockwise",
        }.get(self.type, self.type.lower())
        out = f"The camera {verb}"
        if self.amplitude:
            out += f" with {self.amplitude} amplitude"
        if self.speed:
            out += f" at {self.speed} speed"
        return out


@dataclass
class SoundEvent:
    """`sync` events belong in the shot body, `ambient` in overall_soundscape. The split is
    made here so the two sections cannot describe the same sound twice."""

    text: str
    layer: str = "sync"              # sync | ambient


@dataclass
class ShotPlan:
    n: int
    start_ms: int
    end_ms: int
    beat: str = ""                   # what changes in this shot — the thing that makes it a shot
    camera: CameraMove | None = None
    subjects: list[str] = field(default_factory=list)
    dialogue: list[DialogueLine] = field(default_factory=list)
    sync_sound: list[str] = field(default_factory=list)
    onscreen_text: list[str] = field(default_factory=list)
    word_target: int = 120
    body: str = ""                   # generated prose, filled at render

    @property
    def cut_timestamp(self) -> str | None:
        return None if self.n == 1 else ms_to_timestamp(self.start_ms)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class LoraChoice:
    id: str
    version: int
    file_sha256: str
    strength_requested: float
    strength_applied: float
    triggers: list[dict[str, Any]] = field(default_factory=list)
    registry_revision: str = ""


@dataclass
class ModeDecision:
    mode: Mode
    confidence: float
    rule_fired: str
    signals: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    asked: bool = False


@dataclass
class Plan:
    """Everything structural, decided deterministically. The model only fills prose slots."""

    mode: Mode
    target: Target
    manifest: list[ManifestEntry]
    subjects: list[SubjectPlan]
    speakers: list[SpeakerPlan]
    shots: list[ShotPlan]
    task_types: list[str]
    style_phrase: str = ""            # comma-form medium+look; rendered per mode
    summary: str = ""                 # generated prose; the [task type] prefix is templated
    ambient_sound: list[str] = field(default_factory=list)
    music: str = ""
    loras: list[LoraChoice] = field(default_factory=list)
    mode_decision: ModeDecision | None = None
    planner_version: str = "1"

    def label_counts(self) -> dict[str, int]:
        out = {"Picture": 0, "Video": 0, "Audio": 0}
        for e in self.manifest:
            kind = {"image": "Picture", "video": "Video", "audio": "Audio"}[e.kind.value]
            out[kind] += 1
        return out

    def total_word_target(self) -> int:
        return sum(s.word_target for s in self.shots)

    def hash(self) -> str:
        return _sha({
            "mode": self.mode.value,
            "target": [self.target.nominal_seconds, self.target.frames, list(self.target.canvas)],
            "manifest": [asdict(m) for m in self.manifest],
            "subjects": [asdict(s) for s in self.subjects],
            "speakers": [asdict(s) for s in self.speakers],
            "shots": [{k: v for k, v in asdict(s).items() if k != "body"} for s in self.shots],
            "task_types": self.task_types,
            "ambient": self.ambient_sound,
            "loras": [asdict(l) for l in self.loras],
            "planner_version": self.planner_version,
        })


# --------------------------------------------------------------------------- stage E

@dataclass
class Finding:
    rule: str
    severity: str                    # ERROR | WARN | INFO
    msg: str

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.rule}: {self.msg}"


@dataclass
class IRDocument:
    ir_version: str
    profile: str
    mode: Mode
    prompt: str
    plan: Plan
    sections: dict[str, str]
    prompt_tokens: int = 0
    findings: list[Finding] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    # First-class, not a log line. A silent fallback produced three generations of identical output
    # and cost hours: the caller could not see that the model's work had been discarded.
    source: str = "written"          # written | enriched | draft
    fallback_reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self.source == "draft" and bool(self.fallback_reason)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render_hash(self) -> str:
        """Sampling identity. A LoRA reaches the output through two channels — its trigger
        changes this text, its weights change the render — so the weights must be keyed too."""
        return _sha({
            "prompt": self.prompt,
            "loras": [(l.id, l.file_sha256, l.strength_applied) for l in self.plan.loras],
            "canvas": list(self.plan.target.canvas),
            "frames": self.plan.target.frames,
        })

    def presentation(self) -> dict[str, Any]:
        """User-facing layer. No mode names, no frame counts, no node names.

        A fallback shows here too. The user does not need to hear "draft", but they must not be
        told a directed brief was produced when it was not.
        """
        t = self.plan.target
        shape = {"16:9": "widescreen", "9:16": "vertical", "1:1": "square"}
        ratio = f"{t.canvas[0]}:{t.canvas[1]}"
        w, h = t.canvas
        label = shape.get("16:9" if abs(w / h - 16 / 9) < 0.06 else
                          ("9:16" if abs(w / h - 9 / 16) < 0.06 else
                           ("1:1" if abs(w / h - 1) < 0.06 else ratio)), ratio)
        bits = [f"{t.nominal_seconds:g} seconds", label]
        if len(self.plan.shots) > 1:
            bits.append(f"{len(self.plan.shots)} shots")
        if any(s.dialogue for s in self.plan.shots):
            bits.append("with dialogue")
        if self.plan.music and self.plan.music != "N/A":
            bits.append("with music")
        for l in self.plan.loras:
            bits.append(f"styled with {l.id}")
        out = {"headline": _headline(self.plan), "details": bits}
        if self.fell_back:
            out["notice"] = ("Built from a simple description rather than a directed one — "
                             "the detailed pass did not pass checks.")
            out["degraded"] = True
        return out


def _headline(plan: Plan) -> str:
    anchors = [m for m in plan.manifest if m.role in
               (Role.FRAME_ANCHOR_FIRST, Role.FRAME_ANCHOR_LAST)]
    if plan.mode in (Mode.I2VA, Mode.FL2VA, Mode.L2VA) or anchors:
        return "Animating your image"
    if any(m.role is Role.EDIT_SOURCE for m in plan.manifest):
        return "Editing your clip"
    if any(m.role is Role.CONTINUATION_SOURCE for m in plan.manifest):
        return "Continuing your clip"
    if plan.manifest:
        return "Building a scene from your references"
    return "Building your scene"
