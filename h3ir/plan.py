"""Stage C: the deterministic planner.

Everything structural is decided here and nowhere else: labels and their order, speaker
IDs, the shot timeline and its cut times, per-shot word budgets, camera moves, the
sound split, retention markers, task types. The model is never asked for any of it.

This is what the four open problems from the harness runs reduce to:

  under-length      -> arithmetic. A total word target is split across shots, so length
                       is the sum of parts rather than something the prompt asks for.
  no timed beats    -> structure. Cut times exist before any prose does, and each shot
                       gets its own generation call with its own span and beat.
  sound duplicated  -> partition. Every sound event is assigned to exactly one of `sync`
                       (shot body) or `ambient` (overall_soundscape); the two renderers
                       receive disjoint lists, so they cannot describe the same sound.
  camera off-vocab  -> template. The camera is chosen from the closed enum as data and
                       rendered canonically, so the vocabulary A/B is a profile flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .grid import Target, rows_per_latent_frame, video_latent_t
from .models import (AUDIO_REFERENCE_ROLES, AssetCard, AssetKind, Brief, CameraMove, DialogueLine,
                     ManifestEntry, Mode, Plan, Role, SWAP_ROLES, ShotPlan, SpeakerPlan,
                     SubjectPlan)

# Words of speech per second at a natural delivery (~155 wpm). Used to stop the planner
# putting more dialogue into a shot than can physically be spoken inside it.
WORDS_PER_SECOND = 2.6

# Mid-range of MiniMax's own practice: their published Ref2VA detailed_description is 336
# words, the spec says 350-500, their T2VA is 251 and their I2VA 533 (an image to inventory).
WORD_TARGETS = {
    Mode.REF2VA: 400,
    Mode.T2VA: 330,
    Mode.I2VA: 460,
    Mode.FL2VA: 380,
    Mode.L2VA: 380,
}
MIN_SHOT_MS = 1500
MIN_SHOT_WORDS = 90
SECONDS_PER_SHOT = 3.4


@dataclass
class ProfileOptions:
    """Every belief we have not yet settled by measurement is a field here, not a branch."""

    name: str = "h3ir/2026-08-a"
    # base-en.txt 2.1: "`S.SS` is the effective video duration formatted to exactly two decimal
    # places". `nominal` stays expressible so the old behaviour can be measured rather than argued
    # about, but it is not what anyone gets by default -- it made the draft and the written brief
    # disagree about which number the instruction line carries.
    s_ss_policy: str = "snapped"          # snapped (effective, per the spec) | nominal
    camera_style: str = "canonical"       # canonical | prose
    field_separator: str = "\n\n"
    word_target_override: int | None = None
    seconds_per_shot: float = SECONDS_PER_SHOT
    max_shots: int = 4
    include_summary: bool = True
    include_retention: bool = True
    trigger_repeat_default: int = 1


# --------------------------------------------------------------------------- manifest

def build_manifest(brief: Brief, target: Target) -> list[ManifestEntry]:
    """Labels in the runtime's emission order, which is the only order that makes them true.

    Verified in comfy_extras/nodes_minimax_h3.py: all ref images, then for each ref video
    its paired soundtrack's <Audio j> label FIRST and then <Video k>, then standalone
    audios. Ordinals are 1-based per type in emission order.
    """
    # A frame anchor's ordinal is a fact about its ROLE, not about the order the caller listed the
    # files in. `MiniMaxH3ImageToVideo.execute` appends first_frame and then last_frame, so the
    # runtime emits "<Picture 1>: " before the opening plate and "<Picture 2>: " before the closing
    # one, and the fl2va instruction line asserts exactly that alignment. A brief that listed its
    # closing plate first therefore published a manifest whose <Picture 1> was the LAST frame,
    # against text saying Picture 1 lands at 0.00 seconds -- the two anchors swapped, and silently,
    # because a base mode has no retention_analysis in which any other line could disagree.
    # Stable, and a no-op for every other role: reference images keep the caller's order, which for
    # a reference brief IS the numbering they will wire.
    anchor_rank = {Role.FRAME_ANCHOR_FIRST: 0, Role.FRAME_ANCHOR_LAST: 1}
    images = sorted((a for a in brief.assets if a.kind is AssetKind.IMAGE),
                    key=lambda a: anchor_rank.get(a.role, 2))
    videos = [a for a in brief.assets if a.kind is AssetKind.VIDEO]
    audios = [a for a in brief.assets if a.kind is AssetKind.AUDIO]

    paired: dict[str, Any] = {}
    standalone = []
    for a in audios:
        if a.paired_video_sha256:
            paired[a.paired_video_sha256] = a
        else:
            standalone.append(a)

    out: list[ManifestEntry] = []
    n_pic = n_vid = n_aud = 0
    slot = 0

    def rows_for_image(a) -> int:
        if not a.px:
            return rows_per_latent_frame(*target.canvas)
        w, h = a.px
        if a.sizing == "max":
            import math
            scale = min(1.0, 2048 / min(w, h))
        else:
            import math
            scale = min(1.0, math.sqrt((target.canvas[0] * target.canvas[1]) / (w * h)))
        tw = max(32, round(w * scale / 32) * 32)
        th = max(32, round(h * scale / 32) * 32)
        return rows_per_latent_frame(tw, th)

    for a in images:
        n_pic += 1
        out.append(ManifestEntry(slot=slot, label=f"<Picture {n_pic}>", kind=a.kind,
                                 sha256=a.sha256, wiring=f"ref_image_{n_pic}", role=a.role,
                                 px=a.px, sizing=a.sizing, rows=rows_for_image(a),
                                 composition=a.composition,
                                 replaces=(a.replaces or "").strip()))
        slot += 1

    for v in videos:
        n_vid += 1
        snd = paired.get(v.sha256)
        if snd is not None:
            n_aud += 1
            out.append(ManifestEntry(slot=slot, label=f"<Audio {n_aud}>", kind=snd.kind,
                                     sha256=snd.sha256, wiring=f"ref_video_audio_{n_vid}",
                                     role=snd.role, seconds=snd.seconds,
                                     paired_with=f"<Video {n_vid}>",
                                     characterisation=(snd.note or "").strip()))
            slot += 1
        v_rows = 0
        if v.frames:
            v_rows = video_latent_t(v.frames) * rows_per_latent_frame(*target.canvas)
        out.append(ManifestEntry(slot=slot, label=f"<Video {n_vid}>", kind=v.kind,
                                 sha256=v.sha256, wiring=f"ref_video_{n_vid}", role=v.role,
                                 seconds=v.seconds, frames=v.frames, rows=v_rows))
        slot += 1

    # The LABEL ordinal counts every audio, paired ones included, because that is the order the
    # runtime emits them in. The WIRING name does not: `ref_audios` and `ref_video_audios` are two
    # separate autogrow lists on the node, each numbered from 1, and the runtime takes a standalone
    # audio's ordinal from its position in its own list rather than from its key. So a brief with one
    # paired soundtrack and one score published `<Audio 2>` -- correct -- riding `ref_audio_2`, which
    # is a socket that does not exist: the score goes in `ref_audio_1`. The label was right and the
    # instruction for wiring it up was wrong, which is worse than either alone.
    n_standalone = 0
    for a in standalone:
        n_aud += 1
        n_standalone += 1
        out.append(ManifestEntry(slot=slot, label=f"<Audio {n_aud}>", kind=a.kind,
                                 sha256=a.sha256, wiring=f"ref_audio_{n_standalone}", role=a.role,
                                 seconds=a.seconds,
                                 characterisation=(a.note or "").strip()))
        slot += 1

    return out


# --------------------------------------------------------------------------- subjects

_ROLE_MARKER = {
    Role.SUBJECT: "fully_preserved",
    Role.ENVIRONMENT: "fully_preserved",
    Role.FRAME_ANCHOR_FIRST: "fully_preserved",
    Role.FRAME_ANCHOR_LAST: "fully_preserved",
    Role.EDIT_SOURCE: "fully_preserved",
    Role.CONTINUATION_SOURCE: "partially_preserved",
    Role.STYLE: "weak_reference",
    Role.STORYBOARD: "weak_reference",
    Role.STRUCTURE: "weak_reference",
    # Both swap roles carry the plate's subject into the clip whole, so the plate's own defined
    # role -- supply this identity -- is fully preserved. What changes is the CLIP, and that is
    # said on the clip's line and on the line of the figure being taken over, not here.
    Role.PLACED_SUBJECT: "fully_preserved",
    Role.REPLACEMENT_SUBJECT: "fully_preserved",
}
# `sfx` is `reference`, not a copy, and that was a real 500 rather than a nicety. R22
# (validate.py) forbids a copy marker for the two roles whose definition IS "a property is
# referenced, not the signal", so the renderer wrote `partially_copy`, the validator rejected it,
# and because this is the deterministic draft the compiler raised instead of falling back. Every
# other place the codebase states what `sfx` means agrees with the rule: the card summary is "a
# sound-effect reference" (analyse.py), the retention note is "its sound texture is referenced"
# (audio_relations, below) and the definition line is "a sound-texture reference for the target
# video" (render.py). ref-en.txt 4.2 defines `reference` as "only timbre, rhythm, music style,
# dialogue content, or sound texture is referenced", which is that sentence exactly. So the marker
# table was the wrong side of the contradiction.
_AUDIO_MARKER = {
    Role.BGM: "partially_copy",
    Role.SFX: "reference",
    Role.VOICE_TIMBRE: "reference",
    # The two roles added for ref-en.txt 2.4's other two reference uses. Same side of the same
    # sentence: 4.2 defines `reference` as "only timbre, rhythm, music style, dialogue content, or
    # sound texture is referenced", which names a music style and a rhythm outright.
    Role.MUSIC_STYLE: "reference",
    Role.BEAT_REFERENCE: "reference",
}


def build_subjects(manifest: list[ManifestEntry],
                   cards: dict[str, AssetCard]) -> list[SubjectPlan]:
    """One subject per reusable visible unit, bound to the grounded labels it comes from.

    Spec rule enforced structurally: an image used only to define a character/scene/style
    never gets a standalone <Picture N> line, it is cited inside the subject's definition.
    Because subject_definitions is templated from this, that class of error cannot occur.
    """
    subjects: list[SubjectPlan] = []
    n = 0
    for e in manifest:
        if e.kind is AssetKind.AUDIO:
            continue
        # A style plate lends its look and a structure clip its camera and cutting; like a
        # storyboard they are planning material, never visible units, so they get no
        # <Subject N> and earn their standalone lines in render.py.
        if e.role in (Role.STORYBOARD, Role.STYLE, Role.STRUCTURE):
            continue
        card = cards.get(e.sha256)
        if card is None:
            continue
        entries = card.subjects or [{"kind": "subject", "descriptor": card.summary or "the subject",
                                     "attributes": []}]
        # An environment plate is ONE place, not an inventory of its parts. Emitting a subject per
        # wall, torch and vine spends the word budget re-listing textures and makes every shot
        # restate the same furniture -- the request treats the corridor as a single setting.
        if e.role is Role.ENVIRONMENT:
            n += 1
            subjects.append(_collapse_environment(f"<Subject {n}>", e, card, entries))
            continue
        # Eight views of one man is one man. If a multi-panel sheet were decomposed the way the
        # corridor was decomposed into torch / wall / vines, the brief would define the same
        # person eight times and spend its whole budget doing it.
        if card.is_reference_sheet and len(entries) > 1:
            n += 1
            subjects.append(_collapse_sheet(f"<Subject {n}>", e, card, entries))
            continue
        for s in entries:
            n += 1
            pose_ok = e.role in (Role.FRAME_ANCHOR_FIRST, Role.FRAME_ANCHOR_LAST)
            base = _ROLE_MARKER.get(e.role, "fully_preserved")
            subjects.append(SubjectPlan(
                label=f"<Subject {n}>",
                kind=s.get("kind", "subject"),
                sources=[e.label],
                descriptor=s.get("descriptor") or "the subject",
                attributes=[a for a in (s.get("attributes") or []) if a][:8],
                pose=[a for a in (s.get("pose") or []) if a][:4],
                pose_licensed=pose_ok,
                retention=base,
            ))
    return subjects


def _collapse_sheet(label: str, entry: ManifestEntry, card: AssetCard,
                    parts: list[dict]) -> SubjectPlan:
    """One subject, described from every view the sheet provides.

    The panels are views of the same person, so their attributes merge; their poses do not carry
    (a sheet's poses are the sheet's). The richest descriptor wins.
    """
    attrs: list[str] = []
    for p in parts:
        for a in (p.get("attributes") or []):
            a = (a or "").strip().rstrip(".")
            if a and a.lower() not in {x.lower() for x in attrs}:
                attrs.append(a)
    descriptor = max((p.get("descriptor") or "" for p in parts), key=len) or "the subject"
    kind = next((p.get("kind") for p in parts if p.get("kind")), "person")
    base = _ROLE_MARKER.get(entry.role, "fully_preserved")
    return SubjectPlan(label=label, kind=kind or "person", sources=[entry.label],
                       descriptor=descriptor.strip(), attributes=attrs[:10],
                       pose=[], pose_licensed=False, retention=base)


def _lower_first(s: str) -> str:
    """Card fields arrive sentence-cased; spliced into a list they must not carry a capital."""
    return s[0].lower() + s[1:] if s and s[0].isupper() and not s[:3].isupper() else s


def _collapse_environment(label: str, entry: ManifestEntry, card: AssetCard,
                          parts: list[dict]) -> SubjectPlan:
    """One scene subject whose attributes are the place's notable features."""
    attrs: list[str] = []
    for p in parts:
        desc = _lower_first((p.get("descriptor") or "").strip().rstrip("."))
        feats = [_lower_first(f.strip().rstrip(".")) for f in (p.get("attributes") or []) if f]
        if desc and feats:
            # A comma, not "with": card features are already phrases ("constructed of rough-hewn
            # blocks"), and gluing "with" in front produced "the wall with constructed of".
            attrs.append(f"{desc} — {', '.join(feats[:2])}")
        elif desc:
            attrs.append(desc)
    if card.lighting:
        attrs.append(_lower_first(card.lighting.strip().rstrip(".")))
    descriptor = (card.environment or card.summary or "the setting").strip().rstrip(".")
    descriptor = descriptor[0].lower() + descriptor[1:] if descriptor else "the setting"
    if not descriptor.startswith(("a ", "an ", "the ")):
        descriptor = "the " + descriptor
    base = _ROLE_MARKER.get(entry.role, "fully_preserved")
    return SubjectPlan(label=label, kind="environment", sources=[entry.label],
                       descriptor=descriptor, attributes=attrs[:8], retention=base)


# Words that identify nobody, so two descriptions sharing only these share nothing. Short and
# closed on purpose: this list IS the whole judgement the matching rule makes, and every word added
# to it is a word a caller can no longer be identified by.
_EMPTY_WORDS = frozenset(
    "a an the this that these those his her its their of in on at to with and or is was are "
    "who whom wearing side".split())


def identifying_words(text: str) -> set[str]:
    """The words in a description that could name somebody, lowercased."""
    return {w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if w and w not in _EMPTY_WORDS}


@dataclass(frozen=True)
class SwapDecision:
    """What one `replacement_subject` picture does, worked out but not yet applied.

    Separated from applying it because two callers need the same answer for different reasons:
    `bind_replacement` marks the subjects, and `compile.check_swap` turns the unresolvable ones
    into a sentence. Deriving it twice is how the refusal and the binding come to disagree.
    """

    picture: str                          # "<Picture 1>"
    incoming: SubjectPlan | None          # the subject the picture defines
    named: str                            # who the caller said it replaces, "" if they did not
    candidates: list[SubjectPlan]         # figures in the clip it could take over from
    target: SubjectPlan | None = None     # the one it does take over from
    synthesise: bool = False              # nobody analysed matches: the named figure is its own
    problem: str = ""                     # none | undefined | unnamed | crowded


def swap_decisions(subjects: list[SubjectPlan],
                   manifest: list[ManifestEntry]) -> list[SwapDecision]:
    """One decision per `replacement_subject` picture, in the order they are labelled.

    The rule for "which figure" is LIKE FOR LIKE first: a replacement picture defines a subject of
    some kind (`person`, `object`, ...), the clip's card yields its own subjects with their kinds,
    and the only reading under which the wiring means anything is that the new subject stands in
    for one of the same kind. The measurement clip yields three subjects -- a man, a hammer, a
    plank -- so binding to "the subjects from the clip" without the kind would be ambiguous on the
    very first brief.

    What the CALLER wrote decides the rest, and it is read in exactly one place: choosing among
    several candidates of the right kind. Everywhere else their words are a description, never a
    query, because the alternative is a query that can silently pick the wrong person:

      one candidate      it binds, named or not. The clip's own reading found exactly one figure
                         of that kind and there is nothing to choose between. When the caller
                         named it, their words become its description, because they are the words
                         the caller can recognise and the analyser only saw three frames.
      several candidates the name picks one, by sharing an identifying word with its description.
                         A unique winner binds, and its description becomes the caller's words.
      not resolvable     a name that fits none of them, or fits several, or has no candidate to
                         fit at all: the caller's words ARE the figure, and a subject is made out
                         of them. Never a refusal, and never the leftover candidate.

    That last row is the owner's ruling and it overturns two refusals this function used to
    produce. The candidate list is the analyser's reading of three sampled frames, and that
    reading is wrong routinely: somebody walks in at second four and is in none of them, two
    people overlap and are written down as one, a figure turned away is described in words nobody
    would use for them. So a refusal keyed on it fires when nothing is ambiguous, against
    descriptions the caller was never shown and cannot argue with. In his words: "I think it's
    wrong enforcing mechanically something derived from looking at 3 frames, there would be a lot
    of misfires ... You looked at the clip, it looked at three frames, and you win."

    What survives is the case where nothing was said at all, which is a blank rather than a
    judgement about somebody's eyes: `unnamed` when no figure of that kind was read and no words
    were given, and `crowded` when several were read and no words were given. Both ask for
    something that is never wrong to give, and the panel asks for it before anything is queued.

    Two rules the ruling did NOT touch, and which the fall-through must not quietly undo: like
    for like by kind, and a figure already taken by an earlier picture is not "the only one". An
    unresolvable name makes its own subject; it never falls through to whoever was left over,
    which would be the pick-what-is-left binding this layer exists to refuse.

    Nothing here mutates, and nothing here raises.
    """
    clip = next((m.label for m in manifest if m.role is Role.EDIT_SOURCE), None)
    pictures = [m for m in manifest if m.role is Role.REPLACEMENT_SUBJECT]
    if clip is None or not pictures:
        return []

    spoken_for: set[str] = set()
    out: list[SwapDecision] = []
    for m in pictures:
        named = (m.replaces or "").strip()
        incoming = next((s for s in subjects if m.label in s.sources), None)
        if incoming is None:
            out.append(SwapDecision(m.label, None, named, [], problem="undefined"))
            continue
        # How crowded the CLIP is, and what is still free, are two different questions and both
        # get asked. A figure left over after an earlier picture took the one it named is not "the
        # only figure of its kind"; binding a description that fits neither of them to whichever
        # one remained is the pick-what-is-left guess this layer exists to refuse.
        of_kind = [s for s in subjects
                   if clip in s.sources and s.kind == incoming.kind and s is not incoming]
        free = [s for s in of_kind if s.label not in spoken_for]
        alone = len(of_kind) == 1 and bool(free)

        def bind(target: SubjectPlan) -> None:
            spoken_for.add(target.label)
            out.append(SwapDecision(m.label, incoming, named, free, target=target))

        if alone:
            # One figure of that kind in the whole clip: there is nothing to choose between, so
            # words that read nothing like the analyser's reading are a description of that figure
            # and never a failed query.
            bind(free[0])
        elif not named:
            out.append(SwapDecision(m.label, incoming, named, free,
                                    problem="unnamed" if not of_kind else "crowded"))
        elif not free:
            out.append(SwapDecision(m.label, incoming, named, free, synthesise=True))
        else:
            wanted = identifying_words(named)
            matches = [c for c in free if wanted & identifying_words(c.descriptor)]
            if len(matches) == 1:
                bind(matches[0])
            else:
                # Nothing matched, or several did. The words stand on their own rather than being
                # sent back as a question, and deliberately NOT bound to whichever candidate is
                # left: the reading they failed against saw three frames, and the caller saw the
                # clip.
                out.append(SwapDecision(m.label, incoming, named, free, synthesise=True))
    return out


def bind_replacement(subjects: list[SubjectPlan],
                     manifest: list[ManifestEntry]) -> list[str]:
    """Mark every figure a `replacement_subject` takes over from. Returns their labels.

    Empty when there is nothing to bind or the decision is a question. Those are refused at intake
    (`compile.check_swap`) precisely so they cannot reach a shipped document: what must never
    happen is a brief that claims a swap and names the wrong figure, and leaving the subjects
    untouched here is the only outcome that asserts nothing.

    Appends to `subjects` for a named figure the analyser never saw, which is the one case where
    this list grows after `build_subjects` has numbered it. The new label continues the numbering,
    and it is added before `build_plan` assigns shots so it is scoped like every other subject.
    """
    bound: list[str] = []
    for d in swap_decisions(subjects, manifest):
        if d.incoming is None or d.problem:
            continue
        target = d.target
        if target is None:
            if not d.synthesise:
                continue
            clip = next((m.label for m in manifest if m.role is Role.EDIT_SOURCE), "")
            target = SubjectPlan(label=f"<Subject {len(subjects) + 1}>", kind=d.incoming.kind,
                                 sources=[clip], descriptor=d.named)
            subjects.append(target)
        elif d.named:
            # The caller's words win over the analyser's for the figure being removed. Both name
            # the same person, one of them from three sampled frames and the other from whoever
            # is looking at the clip, and the caller's is the one they will recognise in the brief.
            target.descriptor = d.named
        # ref-en.txt 4.1: "`attribute_transfer` | Referenced characteristics are transferred to a
        # different identifiable target subject". The characteristics are this figure's position in
        # frame, actions and timing; the different identifiable target subject is the one the caller
        # attached the picture for. Build-log 43 deleted the compiler's previous use of this marker
        # because restyling one man keeps the same identifiable subject and the marker never applied.
        # A replacement is the case it was written for, and the only one in this compiler.
        target.retention = "attribute_transfer"
        target.taken_over_by = d.incoming.label
        bound.append(target.label)
    return bound


def build_speakers(brief: Brief, subjects: list[SubjectPlan],
                   manifest: list[ManifestEntry]) -> list[SpeakerPlan]:
    """(SN) in order of first actual vocal event in the TARGET video, contiguous from 1."""
    voice_refs = [m.label for m in manifest if m.role is Role.VOICE_TIMBRE]
    people = [s for s in subjects if s.kind in ("person", "character", "subject")]
    out: list[SpeakerPlan] = []
    seen: dict[str, SpeakerPlan] = {}
    for line in brief.dialogue:
        key = line.speaker_hint or "default"
        if key in seen:
            continue
        sid = f"(S{len(out) + 1})"
        # Caller-owned subject binding (director doctrine, cf. camera_phrase): an explicit
        # 1-based speaker_subject binds the line to that <Subject N>; absent it, keep the
        # positional behavior. Without this a single dialogue line can never move off
        # people[0], which made two-person speaker tests structurally contradictory.
        if line.speaker_subject and 1 <= line.speaker_subject <= len(people):
            subj = people[line.speaker_subject - 1].label
        else:
            subj = people[len(out)].label if len(out) < len(people) else (
                people[0].label if people else None)
        sp = SpeakerPlan(sid=sid, subject=subj,
                         voice_ref=voice_refs[0] if voice_refs else None,
                         onscreen=not line.voiceover,
                         descriptor=line.speaker_hint or "")
        seen[key] = sp
        out.append(sp)
    return out


# --------------------------------------------------------------------------- timeline

def shot_count(target: Target, brief: Brief, mode: Mode, opts: ProfileOptions,
               manifest: list[ManifestEntry]) -> int:
    if brief.shots:
        # A pin is a contract, checked at intake (1..PINNED_SHOTS_MAX and it fits the render), so
        # clamping it to the profile ceiling here would silently break the number the caller set.
        return max(1, brief.shots)
    # Editing a source video preserves its structure; do not invent cuts into it.
    if any(m.role in (Role.EDIT_SOURCE, Role.CONTINUATION_SOURCE) for m in manifest):
        return 1
    # FL2VA interpolates between two frames and the spec says it favours a single shot.
    if mode is Mode.FL2VA:
        return 1
    n = round(target.effective_seconds / opts.seconds_per_shot)
    n = max(1, min(opts.max_shots, n))
    while n > 1 and (target.effective_seconds * 1000) / n < MIN_SHOT_MS:
        n -= 1
    return n


def allocate_from_proposal(target: Target, proposed, mode: Mode,
                           opts: ProfileOptions) -> list[ShotPlan]:
    """Build the timeline from the model's edit. Cut times are its decision; the word budget is
    still arithmetic, split by each shot's real share of the duration."""
    from .shots import spans

    total_words = opts.word_target_override or WORD_TARGETS.get(mode, 400)
    total_ms = int(round(target.effective_seconds * 1000)) or 1
    out: list[ShotPlan] = []
    for i, ((start, end), sh) in enumerate(zip(spans(proposed, target), proposed), start=1):
        share = (end - start) / total_ms
        out.append(ShotPlan(
            n=i, start_ms=start, end_ms=end,
            beat=(f"{sh.purpose}: {sh.action}"
                  + (f" — what changes: {sh.what_changes}" if sh.what_changes else "")).strip(),
            camera=CameraMove(type=sh.camera["type"], amplitude=sh.camera.get("amplitude"),
                              speed=sh.camera.get("speed")) if sh.camera.get("type") else None,
            subjects=list(sh.subjects),
            sync_sound=list(sh.sync_sound),
            word_target=max(MIN_SHOT_WORDS, int(round(total_words * share)))))
    return out


def allocate_shots(target: Target, n_shots: int, mode: Mode,
                   opts: ProfileOptions) -> list[ShotPlan]:
    """Even spacing, used only by the deterministic draft.

    This is arithmetic, and that is the point: the draft has no model, so it cannot plan. When a
    model IS available the timeline comes from allocate_from_proposal instead -- dividing the
    duration and calling the result a plan is what produced two identical shots.
    """
    total_ms = int(target.effective_seconds * 1000)
    # Round cut times to 100 ms so they read like a human edit rather than a division result.
    bounds = [0]
    for i in range(1, n_shots):
        raw = total_ms * i / n_shots
        bounds.append(max(bounds[-1] + MIN_SHOT_MS, int(round(raw / 100.0)) * 100))
    bounds.append(total_ms)
    # If rounding pushed a boundary past the end, walk them back.
    for i in range(len(bounds) - 2, 0, -1):
        if bounds[i] >= bounds[i + 1] - MIN_SHOT_MS:
            bounds[i] = bounds[i + 1] - MIN_SHOT_MS

    total_words = opts.word_target_override or WORD_TARGETS.get(mode, 400)
    shots: list[ShotPlan] = []
    for i in range(n_shots):
        start, end = bounds[i], bounds[i + 1]
        share = (end - start) / total_ms if total_ms else 1.0
        shots.append(ShotPlan(n=i + 1, start_ms=start, end_ms=end,
                              word_target=max(MIN_SHOT_WORDS, int(round(total_words * share)))))
    return shots


def assign_dialogue(shots: list[ShotPlan], lines: list[DialogueLine]) -> None:
    """Greedy fill by physical speech capacity, so no shot is given more words than can be
    spoken inside it. Overflow lands in the last shot rather than being dropped."""
    if not lines:
        return
    caps = [max(1.0, (s.duration_ms / 1000.0) * WORDS_PER_SECOND) for s in shots]
    used = [0.0] * len(shots)
    cursor = 0
    for line in lines:
        w = len(line.text.split())
        target = None
        # forward-first: keep speech in narrative order where it fits
        for i in range(cursor, len(shots)):
            if used[i] + w <= caps[i]:
                target = i
                break
        if target is None:
            # nothing fits: give it to whichever shot has the most room left, rather than
            # piling every overflow line onto the last shot
            target = max(range(len(shots)), key=lambda i: caps[i] - used[i])
        shots[target].dialogue.append(line)
        used[target] += w
        cursor = target


def split_sound(events: list[dict[str, Any]]) -> tuple[dict[int, list[str]], list[str]]:
    """Partition sound into per-shot synchronized events and whole-video ambience.

    A sound belongs to exactly one side. That is why the two rendered sections cannot
    duplicate each other -- it is not an instruction to the model, it is set arithmetic.
    """
    sync: dict[int, list[str]] = {}
    ambient: list[str] = []
    for e in events:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        if e.get("layer") == "ambient" or not e.get("shot"):
            if text not in ambient:
                ambient.append(text)
        else:
            sync.setdefault(int(e["shot"]), []).append(text)
    return sync, ambient


# --------------------------------------------------------------------------- task types

def derive_task_types(manifest: list[ManifestEntry], brief: Brief) -> list[str]:
    """From roles, never from prose. Order follows the spec's own table order."""
    types: list[str] = []
    roles = {m.role for m in manifest}
    has_audio = any(m.kind is AssetKind.AUDIO for m in manifest)

    if roles & {Role.FRAME_ANCHOR_FIRST, Role.FRAME_ANCHOR_LAST}:
        types.append("keyframe completion")
    # The two swap roles are in here for the reason ref-en.txt 3 gives `reference generation`:
    # the picture "provides generation guidance for a character ... without serving as a concrete
    # frame or as the source video being edited or continued". The clip is what is edited; the
    # picture is guidance, and both types are true at once, which is what ` + ` is for.
    if roles & {Role.SUBJECT, Role.ENVIRONMENT, Role.STYLE, Role.STORYBOARD, Role.STRUCTURE,
                Role.VOICE_TIMBRE, *SWAP_ROLES}:
        types.append("reference generation")
    if Role.EDIT_SOURCE in roles:
        types.append("video editing")
    if Role.CONTINUATION_SOURCE in roles:
        types.append("video continuation")
    if has_audio:
        # The task type has to agree with the retention marker the same role produces, because
        # ref-en.txt 3 defines the two against each other: `audio reuse` is "the same audio signal
        # is reused in full or in part" and `audio reference` is "the signal is not copied
        # directly; only its music style, timbre, dialogue or lyric content, sound-effect texture,
        # beat, or continuity is referenced". `bgm` copies part of the signal, so it is reuse; a
        # timbre reference, a sound-effect texture, a music style and a beat are all named in that
        # second sentence, so they are not. A summary claiming reuse over a retention line saying
        # `reference` is the document contradicting itself, and M15 now catches that shape.
        if any(m.role is Role.BGM for m in manifest):
            types.append("audio reuse")
        if any(m.role in AUDIO_REFERENCE_ROLES for m in manifest):
            types.append("audio reference")
    out: list[str] = []
    for t in types:
        if t not in out:
            out.append(t)
    return out or ["reference generation"]


def taken_definition(s: SubjectPlan) -> str:
    """The definition line for the figure a replacement takes over from.

    Its ATTRIBUTES are deliberately dropped, and that is the whole point of this function rather
    than a flag. Measured on the first render of the role: the ordinary form spelled out "a thick
    full dark brown beard, short dark brown hair, a red and black plaid flannel shirt worn open,
    a grey crew-neck t-shirt underneath" for a man the caller had asked to remove -- and H3 drew
    him, holding the frame for the opening moments before the replacement took over. Nothing cited
    him in detailed_description in any of four seeds; the identity list in subject_definitions was
    enough on its own, because that section is conditioning like every other.

    So this line carries only what identifies him IN THE SOURCE -- the descriptor, and the label he
    came from -- plus the one sentence that says he is not in the target video. It is the style
    plate's lesson and the structure clip's, for a subject: a label whose contents must stay out of
    the render must not be described as something to draw.

    Shared with `prose._definition_lines` rather than written twice, because the writer is handed
    these lines to use or reword and a second copy would drift from the draft's.
    """
    src = ", ".join(s.sources)
    return (f"{s.label} is {s.descriptor} in {src}, the figure {s.taken_over_by} replaces. "
            f"{s.label} does NOT appear in the target video: only its position in frame, its "
            f"actions and its timing carry over to {s.taken_over_by}.")


def retention_note(s: SubjectPlan) -> str:
    """What must survive is the identity, never the plate's pose."""
    # The one subject whose note is not about survival. `attribute_transfer` is ref-en.txt 4.1's
    # "referenced characteristics are transferred to a different identifiable target subject", so
    # the note has to name the characteristics that move, name where they land, and say that this
    # figure's own appearance does not survive -- otherwise the marker and the sentence under it
    # describe two different relationships, which is the failure mode that validates cleanly.
    if s.retention == "attribute_transfer" and s.taken_over_by:
        return (f"{s.taken_over_by} takes over the position in frame, the actions and the timing "
                f"of {s.descriptor}, whose own appearance is not retained")
    if not s.attributes:
        return f"{s.descriptor} is retained"
    attrs = ", ".join(s.attributes[:5])
    lead = "" if re.match(r"(?:the|a|an)\b", attrs, re.I) else "the "
    return (f"{lead}{attrs} are retained" if len(s.attributes) > 1
            else f"{lead}{attrs} is retained")


# --------------------------------------------------------------------------- entry point

def build_plan(brief: Brief, mode: Mode, cards: dict[str, AssetCard], *,
               proposal=None,
               beats: list[dict[str, Any]] | None = None,
               sound_events: list[dict[str, Any]] | None = None,
               style_phrase: str = "", music: str = "",
               opts: ProfileOptions | None = None,
               loras: list | None = None,
               mode_decision=None) -> Plan:
    opts = opts or ProfileOptions()
    target = Target.build(brief.seconds, brief.aspect, brief.canvas, brief.megapixels)
    manifest = build_manifest(brief, target)
    subjects = build_subjects(manifest, cards)
    # Before the notes are written, because `retention_note` reads the marker this sets.
    bind_replacement(subjects, manifest)
    speakers = build_speakers(brief, subjects, manifest)

    if proposal is not None:
        shots = allocate_from_proposal(target, proposal, mode, opts)
    else:
        n = shot_count(target, brief, mode, opts, manifest)
        if beats:
            n = max(1, min(len(beats), brief.shots or opts.max_shots))
        shots = allocate_shots(target, n, mode, opts)
    assign_dialogue(shots, brief.dialogue)

    sync, ambient = split_sound(sound_events or [])
    for s in shots:
        s.sync_sound = sync.get(s.n, [])

    if beats and proposal is None:
        for s, b in zip(shots, beats):
            s.beat = (b.get("beat") or "").strip()
            cam = b.get("camera") or {}
            if cam.get("type"):
                s.camera = CameraMove(type=cam["type"], amplitude=cam.get("amplitude"),
                                      speed=cam.get("speed"))
            named = [x for x in (b.get("subjects") or []) if x]
            s.subjects = named or [x.label for x in subjects]
    elif proposal is None:
        for s in shots:
            s.subjects = [x.label for x in subjects]
    for s in shots:
        s.subjects = [x for x in s.subjects if x in {y.label for y in subjects}] \
            or [y.label for y in subjects]

    # Which shots each subject appears in — needed for the mandated '(appears in [Shot N])'.
    for subj in subjects:
        subj.appears_in = [s.n for s in shots if subj.label in s.subjects] or [s.n for s in shots]
        subj.retention_note = retention_note(subj)

    # On-screen text goes to the first shot unless a beat claimed it.
    if brief.onscreen_text and shots:
        shots[0].onscreen_text = list(brief.onscreen_text)

    return Plan(mode=mode, target=target, manifest=manifest, subjects=subjects,
                speakers=speakers, shots=shots,
                task_types=derive_task_types(manifest, brief),
                style_phrase=style_phrase.strip(),
                ambient_sound=ambient,
                music=(music or "").strip() or ("N/A" if brief.silent else ""),
                loras=list(loras or []), mode_decision=mode_decision)


def audio_relations(plan: Plan) -> list[tuple[str, str, str]]:
    """(label, marker, note) for every attached audio, from its role."""
    out = []
    for m in plan.manifest:
        if m.kind is not AssetKind.AUDIO:
            continue
        marker = _AUDIO_MARKER.get(m.role, "reference")
        if m.role is Role.VOICE_TIMBRE:
            note = (f"the target speaker follows {m.label}'s voice timbre and delivery "
                    "without copying the original signal")
        elif m.role is Role.BGM:
            note = f"the background music from {m.label} is reused beneath the new audio"
        elif m.role is Role.MUSIC_STYLE:
            # Says what is taken AND that the signal is not, in one sentence, because a note that
            # only names the property leaves the copy question open for the next reader.
            note = (f"the target video's score follows {m.label}'s instrumentation and tempo and "
                    "is newly generated, without copying the original signal")
        elif m.role is Role.BEAT_REFERENCE:
            note = (f"{m.label}'s beat sets the timing of the cuts and the action, without copying "
                    "the original signal")
        else:
            note = f"{m.label}'s sound texture is referenced"
        out.append((m.label, marker, note))
    return out
