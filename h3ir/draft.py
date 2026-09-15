"""The deterministic draft: a complete, valid IR with no prose model involved.

This inverts the usual arrangement, and the inversion is the point. In a sysprompt-first
design the model's output *is* the product, so every model failure is a product failure. Here
the draft is the **product floor** and the LLM pass is a strictly additive prose enrichment
over a bounded surface. Downstream code consumes one shape either way.

Three things follow:

  * A leaked reasoning block, a truncated reply, or a validator error is no longer an outage.
    It is a fallback, and the caller still gets something valid.
  * `compile_brief(..., llm=False)` is a free A/B baseline. If the enriched brief cannot beat
    its own no-LLM draft, the prose pass is not earning its place.
  * The draft is deterministic, so if it ever fails validation that is OUR bug and it raises
    rather than being papered over.

The draft is deliberately thin, not fake. It says only what the wiring and the asset cards
actually support, which is why it under-writes against the word target and says so.
"""
from __future__ import annotations

import re
from typing import Any

from .models import AssetCard, AssetKind, Brief, Mode, ShotPlan, SubjectPlan
from .plan import ProfileOptions, _lower_first, build_plan
from .style import _tidy, resolve_style

# A deterministic camera rotation: establish, then vary. Chosen from the closed vocabulary so
# the draft is on-vocabulary for the same reason the enriched version is.
DRAFT_CAMERA = [
    {"type": "Push In", "amplitude": "small", "speed": "slow"},
    {"type": "Truck Right", "amplitude": "small", "speed": "slow"},
    {"type": "Static Shot", "amplitude": None, "speed": None},
    {"type": "Pull Out", "amplitude": "small", "speed": "slow"},
]

# The floor honours the dial. The draft is the product floor rather than a degraded mode, so it has
# to satisfy every setting the product offers -- and it did not: at `extreme` its small/slow rotation
# tripped Q3, the draft failed its own validator, and the compile RAISED. A user-facing setting
# turned into a crash. The raise was correct and caught it on the first run; exempting the draft from
# the check would have been the wrong fix, because a floor that quietly ignores the setting is the
# silent degradation this service refuses.
#
# `Static Shot` keeps its Nones on purpose: amplitude and speed are meaningless for a camera that
# does not move, and inventing values for it would be on-vocabulary nonsense. So the rotation at
# maximal magnitude drops it -- a held frame is not the boldest available option.
_MAXIMAL_CAMERA = [
    {"type": "Push In", "amplitude": "large", "speed": "fast"},
    {"type": "Truck Right", "amplitude": "large", "speed": "fast"},
    {"type": "Pull Out", "amplitude": "large", "speed": "fast"},
    {"type": "Tilt Up", "amplitude": "large", "speed": "fast"},
]


# `assertive` commits on ONE dimension -- large amplitude, unhurried. Nothing CHECKS the camera at
# `bold`, and the rotation stays anyway: a fallback that ignores the setting the caller asked for is
# the silent degradation this service refuses everywhere else. Enforcement is not the reason the floor
# is faithful; being the floor is.
_ASSERTIVE_CAMERA = [
    {"type": "Push In", "amplitude": "large", "speed": None},
    {"type": "Truck Right", "amplitude": "large", "speed": None},
    {"type": "Static Shot", "amplitude": None, "speed": None},
    {"type": "Pull Out", "amplitude": "large", "speed": None},
]


def draft_camera(magnitude: str) -> list[dict]:
    """The rotation for a magnitude.

    `plain` and `measured` share one rotation, because the request carries no instruction to lean on
    there and inventing a difference would make the floor lie about which setting produced it.

    **A director does not reach this, and that is deliberate.** An earlier version narrowed the
    rotation to a profile's own camera vocabulary, which meant a profile mechanically decided a
    structural field in the deterministic floor. The owner settled the shape of a profile the other
    way: "not mechanically enforced, just steered". A profile is prose in the ask, so it steers the
    WRITTEN brief and leaves the floor alone. The dial still reaches here, because the dial is a
    setting with an enforced meaning at both ends and a fallback that ignored it would be the silent
    degradation this service refuses everywhere else."""
    return {"maximal": _MAXIMAL_CAMERA, "assertive": _ASSERTIVE_CAMERA}.get(magnitude, DRAFT_CAMERA)

DRAFT_BEATS = ["the scene is established",
               "the action develops",
               "the movement continues",
               "the action settles"]


def _style_from_cards(brief: Brief, cards: dict[str, AssetCard]) -> str:
    """Delegates to the shared resolver. The draft and the enriched path must not have separate
    opinions about style -- the draft is the fallback, and a fallback that renders a different
    style from the thing it replaces is worse than no fallback."""
    return resolve_style(brief, cards).phrase


def _environment(cards: dict[str, AssetCard]) -> str:
    for c in cards.values():
        if c.environment:
            return _tidy(c.environment)
    return ""


def _subject_clause(subjects: list[SubjectPlan], use_labels: bool) -> str:
    """`use_labels` is mode-derived, not cosmetic. <Subject N> means something only because
    subject_definitions defines it, and base modes have no such section -- so a label there is
    a pointer to nothing. Same rule as the prose path; this is the second place it applies."""
    if not subjects:
        return ""
    bits = []
    for s in subjects[:3]:
        attrs = ", ".join(s.attributes[:4])
        head = f"{s.label}, {s.descriptor}" if use_labels else s.descriptor
        bits.append(head + (f" with {attrs}" if attrs else ""))
    if len(bits) == 1:
        return bits[0]
    return ", and ".join([", ".join(bits[:-1]), bits[-1]]) if len(bits) > 2 else " and ".join(bits)


def _anchor_path(subjects: list[SubjectPlan], brief: Brief, environment: str) -> list[str]:
    """FL2VA's body: the path from the first frame to the last one.

    Two plates are not two subjects standing in one frame, they are the same shot at its two ends,
    and base-en.txt 3.2 rules out the alternative in as many words -- "The body should not repeat
    two static image descriptions; instead, it should supply the motion path that connects them."
    The floor used to open on both at once ("The frame holds <the showroom> and <the car>"), which
    asserts the closing plate's content at 0.00 seconds, against a keyframe latent that pins the
    opposite.

    Still says only what is supported: each plate's content comes from its own card, and the change
    between them is the request in the caller's own words. Structure per 3.2 -- first-frame state,
    observable change, last-frame state -- with the camera in the middle where the spec's own
    example puts it. <Picture N> is deliberately BARE here: that is this mode's notation (2.1) and
    the form its published example uses throughout.
    """
    opening = [s for s in subjects if "<Picture 1>" in s.sources]
    closing = [s for s in subjects if "<Picture 2>" in s.sources]
    parts: list[str] = []

    head = "The shot opens in the composition established by Picture 1"
    clause = _subject_clause(opening, use_labels=False)
    parts.append(f"{head}, holding {_lower_first(clause)}" if clause else head)
    if environment:
        parts.append(f"The setting is {_lower_first(environment)}")
    parts.append("{{CAM}}")
    intent = brief.intent.strip().rstrip(".")
    if intent:
        parts.append(f"Across the shot, {_lower_first(intent)}")
    tail = ("The framing narrows toward the pose, spacing and composition established by "
            "Picture 2, which the shot settles into at the end")
    landing = _subject_clause(closing, use_labels=False)
    parts.append(f"{tail}, holding {_lower_first(landing)}" if landing else tail)
    return parts


def draft_shot_body(shot: ShotPlan, subjects: list[SubjectPlan], brief: Brief,
                    environment: str, mode: Mode) -> str:
    """Says only what is supported. Placeholders are emitted so the renderer's substitution
    path is identical for a draft and for an enriched brief -- one code path, one set of bugs."""
    present = [s for s in subjects if s.label in shot.subjects] or subjects
    use_labels = mode is Mode.REF2VA
    parts: list[str] = []

    if mode is Mode.FL2VA and shot.n == 1:
        parts += _anchor_path(present, brief, environment)
    elif shot.n == 1:
        if present:
            parts.append(f"The frame holds {_subject_clause(present, use_labels)}")
        else:
            parts.append(f"The frame holds the scene described as: {brief.intent.strip()}")
        if environment:
            env = environment[0].lower() + environment[1:] if environment else environment
            parts.append(f"The setting is {env}")
        parts.append("{{CAM}}")
    else:
        # base-en.txt 4.2 closes the cut phrase to five forms and "the framing changes" is not one of
        # them. It is our own floor's wording, so every fallback shipped a shot boundary off the
        # vocabulary the model was trained on: 22 of the 152 timestamped openings measured across the
        # re-fired corpus, and all but a handful were this sentence. The draft is the product floor,
        # which makes it the last place that should be paraphrasing a closed set.
        if present:
            parts.append(f"The shot cuts to {_subject_clause(present, use_labels)}, "
                         "still in frame")
        else:
            parts.append("The shot cuts to another view of the same scene")
        parts.append("{{CAM}}")

    for snd in shot.sync_sound[:2]:
        parts.append(snd.rstrip("."))
    for i in range(1, len(shot.dialogue) + 1):
        parts.append(f"{{{{D{i}}}}}")
    for t in shot.onscreen_text[:2]:
        parts.append(f'Text reading "{t}" is visible in frame')

    return ". ".join(p.rstrip(".") for p in parts if p).strip()


def deterministic_draft(brief: Brief, mode: Mode, cards: dict[str, AssetCard], *,
                        opts: ProfileOptions | None = None, loras=None, mode_decision=None,
                        licence=None, camera_phrase: dict | None = None):
    """A complete Plan with every prose slot filled and no prose model called."""
    opts = opts or ProfileOptions()
    if licence is None:
        from .licence import resolve_licence
        licence = resolve_licence(brief, cards)
    skeleton = build_plan(brief, mode, cards, opts=opts, loras=loras,
                          mode_decision=mode_decision)

    from .creativity import MAGNITUDE, parse
    # Controlled experiment (director ruling 2026-09-14): the caller's camera_phrase
    # replaces the rotation wholesale; production (None) keeps the template behavior.
    if camera_phrase is not None:
        rotation = [dict(camera_phrase)]
    else:
        rotation = draft_camera(MAGNITUDE[parse(brief.creativity)])

    beats: list[dict[str, Any]] = []
    for i, _ in enumerate(skeleton.shots):
        beats.append({
            "beat": DRAFT_BEATS[min(i, len(DRAFT_BEATS) - 1)],
            "camera": dict(rotation[i % len(rotation)]),
            "subjects": [s.label for s in skeleton.subjects],
            "sync_sound": [],
        })

    # An audio card's `summary` is PROVENANCE -- "a spoken vocal reference supplying voice timbre and
    # delivery, described by the caller as: ... (6.00s)". Appending it here put that whole string into
    # overall_soundscape, which the spec reserves for the target video's ambience and physical sound.
    # The caller's characterisation now reaches subject_definitions instead, where the label is
    # defined, and the soundscape is left to describe the video.
    ambient: list[str] = []
    if not ambient:
        ambient = ["Room tone continues throughout the video."]

    plan = build_plan(brief, mode, cards, beats=beats,
                      sound_events=[{"text": a, "layer": "ambient"} for a in ambient],
                      style_phrase=resolve_style(brief, cards, licence).phrase,
                      music="N/A",
                      opts=opts, loras=loras, mode_decision=mode_decision)

    subs = ", ".join(s.label for s in plan.subjects)
    plan.summary = (f"The target video shows {subs} in the scene described by the request."
                    if subs else "The target video shows the scene described by the request.")

    env = _environment(cards)
    for shot in plan.shots:
        shot.body = draft_shot_body(shot, plan.subjects, brief, env, mode)
    return plan
