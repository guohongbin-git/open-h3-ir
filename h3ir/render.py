"""Stage D (second half): the deterministic renderer.

No model is called here. Given a Plan whose prose slots are filled, this produces the
exact prompt string, and re-rendering the same Plan must produce byte-identical output.

Placeholder substitution is the mechanism that keeps the model away from anything that
must be exact. The prose stage is asked to emit `{{CAM}}` and `{{D1}}` tokens; this module
replaces them with canonical camera phrasing and with the user's dialogue markup. So the
user's words never pass through the model, and the camera vocabulary is a template
decision rather than something we hope the model remembers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .grid import instruction_line_for, ms_to_timestamp
from .models import (AssetKind, DialogueLine, Mode, Plan, Role, SWAP_ROLES, SpeakerPlan,
                     SubjectPlan)
from .plan import ProfileOptions, audio_relations, taken_definition
from .textnorm import sentences

CAM_TOKEN = "{{CAM}}"
DLG_TOKEN = re.compile(r"\{\{D(\d+)\}\}")


@dataclass
class RenderResult:
    prompt: str
    sections: dict[str, str]
    notes: list[str]
    # The per-shot text as it actually shipped, after placeholder substitution and repairs. Checks
    # that ask "does this shot name its subject" must read THIS, not the pre-render draft.
    shot_bodies: dict[int, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- pieces

def instruction_line(plan: Plan, opts: ProfileOptions) -> str:
    """Delegates to grid.instruction_line_for, which the repair pass and the validator also use."""
    return instruction_line_for(plan.mode.value, plan.shots[-1].n if plan.shots else 1,
                                plan.target.s_ss(opts.s_ss_policy))


def _swap_clause(plan: Plan, s: SubjectPlan) -> str:
    """The reference role of a swap subject, in the definition line where ref-en.txt 2 wants it.

    "explain what its label denotes, its reference role, and the main features to follow" -- and
    for these two roles the reference role IS the whole point, because the identity is what an
    ordinary `subject` already supplies and the relationship to the clip is what it does not.
    """
    roles = {m.label: m.role for m in plan.manifest}
    clip = next((m.label for m in plan.manifest if m.role is Role.EDIT_SOURCE), "")
    role = next((roles.get(src) for src in s.sources if roles.get(src) in SWAP_ROLES), None)
    if not clip or role is None:
        return ""
    if role is Role.PLACED_SUBJECT:
        return (f" {s.label} is placed into {clip} in the target video, which is otherwise "
                "unchanged.")
    taken = next((x.label for x in plan.subjects if x.taken_over_by == s.label), "")
    if not taken:
        return ""
    return (f" {s.label} takes the place of {taken} in {clip}, keeping that figure's position in "
            "frame, actions and timing.")


def render_subject_definitions(plan: Plan) -> str:
    """Templated, which is why a redundant standalone <Picture N> source line cannot occur."""
    lines: list[str] = []
    for s in plan.subjects:
        # The figure being replaced gets a line that identifies it and nothing to draw. See
        # plan.taken_definition for the render that made that necessary.
        if s.taken_over_by:
            lines.append(taken_definition(s))
            continue
        src = ", ".join(s.sources)
        # IDENTITY only. A plate's stance, gesture and expression are properties of that
        # photograph, and the request owns what the subject does -- unless the plate IS a frame
        # of the video, in which case its pose is the video's pose and carries forward.
        facts = list(s.attributes)
        if s.pose_licensed and s.pose:
            facts += list(s.pose)
        attrs = ", ".join(facts)
        swap = _swap_clause(plan, s)
        if attrs:
            lines.append(f"{s.label} is {s.descriptor} in {src}, with {attrs}.{swap}")
        else:
            lines.append(f"{s.label} is {s.descriptor} in {src}.{swap}")
    for m in plan.manifest:
        if m.kind is AssetKind.VIDEO and m.role is Role.EDIT_SOURCE:
            lines.append(f"{m.label} is the source video for the target video edit.")
        elif m.kind is AssetKind.VIDEO and m.role is Role.CONTINUATION_SOURCE:
            lines.append(f"{m.label} is the source video the target video continues from.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.STORYBOARD:
            # A storyboard is a shot-planning anchor, not a reusable visible unit, so
            # `build_subjects` skips it and it gets no <Subject N> -- which left a storyboard-only
            # ref2va brief with an EMPTY subject_definitions and an uncited <Picture N>: two
            # ERRORs in the deterministic draft, so the compile raised and the caller got a bare
            # 500. ref-en.txt 2.2 gives this label its own line and its own form ("<Picture 3> is
            # a storyboard reference for [Shot 1] and [Shot 2], defining their viewpoint, subject
            # placement, and shot order"), which is the one case where a standalone <Picture N>
            # line is what the spec asks for. It earns that line by having a retention entry, so
            # L5 is satisfied by render_retention below rather than by an exemption here.
            shots = " and ".join(f"[Shot {s.n}]" for s in plan.shots) or "[Shot 1]"
            lines.append(f"{m.label} is a storyboard reference for {shots}, defining their "
                         "viewpoint, subject placement, and shot order.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.STYLE:
            # The same standalone-line logic as the storyboard above: `build_subjects` skips a
            # style plate so its contents cannot walk into the video, and this line is the one
            # form the spec gives a style-only picture. R29 holds the written path to it.
            lines.append(f"{m.label} is the style and composition reference for the target video, "
                         "defining its medium, line, palette, shading and composition.")
        elif m.kind is AssetKind.VIDEO and m.role is Role.STRUCTURE:
            # The structure sibling: the clip lends how it is shot and cut, never what is in it.
            # R30 holds the written path to this form.
            lines.append(f"{m.label} is the source video providing the camera movement and "
                         "cutting rhythm for the target video.")
    for m in plan.manifest:
        if m.kind is not AssetKind.AUDIO:
            continue
        sid = next((sp.sid for sp in plan.speakers if sp.voice_ref == m.label), None)
        subj = next((sp.subject for sp in plan.speakers if sp.voice_ref == m.label), None)
        # The caller's characterisation goes HERE, in the definition, because H3's tokenizer emits
        # only "<Audio j>: " and never the signal -- so this text is the sole channel by which the
        # conditioning encoder learns what the audio is. "containing a spoken vocal layer" told it
        # nothing at all.
        said = m.characterisation
        detail = f" — {said}" if said else ""
        if m.role is Role.VOICE_TIMBRE and sid:
            who = f"{subj} {sid}" if subj else f"the speaker {sid}"
            lines.append(f"{m.label} is the voice-timbre reference for {who}, "
                         f"containing a spoken vocal layer{detail}.")
        elif m.role is Role.BGM:
            paired = f" of {m.paired_with}" if m.paired_with else ""
            lines.append(f"{m.label} is the synchronized audio track{paired}, "
                         f"providing the background music{detail}.")
        elif m.role is Role.MUSIC_STYLE:
            # ref-en.txt 2.4: "An `<Audio N>` definition primarily states the audio's role." The
            # role here is a style reference, so the line names the style and says the score is new
            # -- the same fact the retention note carries, in the section that introduces the label.
            lines.append(f"{m.label} is a music-style reference for the target video's newly "
                         f"generated score{detail}.")
        elif m.role is Role.BEAT_REFERENCE:
            lines.append(f"{m.label} is a beat reference whose rhythm sets the timing of the "
                         f"target video's cuts and action{detail}.")
        else:
            lines.append(f"{m.label} is a sound-texture reference for the target "
                         f"video{detail}.")
    return "\n".join(lines)


def render_retention(plan: Plan) -> str:
    lines: list[str] = []
    for s in plan.subjects:
        shots = ", ".join(f"[Shot {n}]" for n in s.appears_in)
        note = s.retention_note
        # `plan.retention_note` already writes the transfer sentence for this marker, naming what
        # moves and where it lands. There used to be a style clause spliced on here, from the
        # reading of `attribute_transfer` that build-log 43 deleted; nothing sets that marker for
        # a restyle any more, and the medium travels through the style opening instead.
        if s.retention == "attribute_transfer" and not s.taken_over_by:
            note = f"{note} — the target subject this transfers to is not named"
        lines.append(f"{s.label} (appears in {shots}): {s.retention} - {note}.")
    for m in plan.manifest:
        if m.kind is AssetKind.VIDEO:
            if m.role is Role.EDIT_SOURCE:
                # Under a swap the clip's line has to enumerate what survives, because what
                # survives IS the request: the caller asked for the same camera, the same motion
                # and the same words with a different figure in them, and a line that says only
                # "while the edit is applied" leaves every one of those unstated.
                swapped = [x for x in plan.subjects if x.taken_over_by]
                placed = [x for x in plan.subjects
                          if any(e.role is Role.PLACED_SUBJECT and e.label in x.sources
                                 for e in plan.manifest)]
                if swapped or placed:
                    # Every change, not the first one. Two pictures can each replace a different
                    # figure, and a line naming one of them describes an edit the wiring does not
                    # perform: the second subject would be defined, marked and then unaccounted for
                    # in the sentence that says what happens to the clip.
                    changes = [f"{x.taken_over_by} takes the place of {x.label}" for x in swapped]
                    changes += [f"{x.label} is placed into the scene" for x in placed]
                    what = (changes[0] if len(changes) == 1
                            else ", ".join(changes[:-1]) + f" and {changes[-1]}")
                    # `fully_preserved` and "are maintained while ... is edited", which is the
                    # shape MiniMax's own README gives for a video being edited: "the original
                    # camera framing, warm golden hour lighting, grassy hill setting, and
                    # background white lambs are maintained while the central character is edited".
                    # Their README and not ref-en.txt: the spec defines the four markers and shows
                    # worked examples for people and objects, and has none for an edited video. It
                    # is the best evidence there is, and it is an example rather than a rule.
                    #
                    # What it replaced said `partially_preserved` -- some characteristics change --
                    # in the same breath as "held exactly", which says none do. It also promised a
                    # frame-for-frame match, which the measurement in AGENTS.md says these weights
                    # do not deliver. The branch below already says almost exactly this; the two
                    # now agree.
                    lines.append(
                        f"{m.label} (source video editing): fully_preserved - the original framing, "
                        "camera movement, lighting, setting, action and timing are maintained "
                        f"while {what}.")
                else:
                    lines.append(f"{m.label} (source video editing): fully_preserved - the original "
                                 "framing, lighting and setting are maintained while the edit is applied.")
            elif m.role is Role.CONTINUATION_SOURCE:
                lines.append(f"{m.label} (continuation source): partially_preserved - the setting "
                             "and subject continue from its final state.")
            elif m.role is Role.STRUCTURE:
                lines.append(f"{m.label} (camera movement and cutting rhythm): weak_reference - "
                             "the camera moves and edit timing are followed, while the video's "
                             "contents are not reproduced.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.FRAME_ANCHOR_FIRST:
            lines.append(f"{m.label} ([Shot 1] first frame): fully_preserved - the composition, "
                         "subject placement and lighting of the opening frame are held.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.FRAME_ANCHOR_LAST:
            lines.append(f"{m.label} ([Shot {plan.shots[-1].n}] last frame): fully_preserved - "
                         "the final composition and subject placement are reached.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.STORYBOARD:
            # The entry that earns the standalone definition line above. `weak_reference` is what
            # the role's own marker table says (plan._ROLE_MARKER) and what the marker means here:
            # the planning information is followed, the drawing itself is not preserved.
            lines.append(f"{m.label} (storyboard reference): weak_reference - the viewpoint, "
                         "subject placement and shot order are followed, while the drawing itself "
                         "is not reproduced.")
        elif m.kind is AssetKind.IMAGE and m.role is Role.STYLE:
            lines.append(f"{m.label} (style and composition): weak_reference - the target video "
                         "adheres to the reference's aesthetic, while its contents are not "
                         "reproduced.")
    for label, marker, note in audio_relations(plan):
        lines.append(f"{label}: {marker} - {note}.")
    return "\n".join(lines)


def style_sentence(phrase: str) -> str:
    """Ref2VA form: one sentence on its own line before [Shot 1]. Built from the same phrase
    the base modes splice inline, so the two modes cannot drift apart."""
    p = (phrase or "").strip().rstrip(".")
    if not p:
        return ""
    head, _, rest = p.partition(",")
    head = head.strip()
    head = head[0].lower() + head[1:] if head else head
    # The medium the model names often already ends in "style" ("anime style"), and appending
    # ours produced "in anime style style". Drop the duplicate rather than the word.
    head = re.sub(r"\s+styles?$", "", head, flags=re.I)
    rest = rest.strip().rstrip(",")
    if not rest:
        return f"The target video is in {head} style."
    # "with" needs a noun phrase after it. A remainder that is only treatment adjectives produced
    # "in anime style with cinematic." -- third grammar bug in this one sentence, so the join is
    # now chosen from the remainder's shape rather than fixed.
    from .style import TREATMENT_TERMS
    items = [x.strip() for x in rest.split(",") if x.strip()]
    # "with" needs a noun phrase directly after it. If the remainder opens with an adjective, a
    # comma is both correct and better reading, whatever follows.
    first_is_adjectival = bool(items) and items[0].lower() in TREATMENT_TERMS
    joiner = ", " if first_is_adjectival else " with "
    return f"The target video is in {head} style{joiner}{', '.join(items)}."


def style_prefix(phrase: str) -> str:
    """Base-mode form: the comma-separated style list the spec's own examples open with
    ("[Shot 1] Live-action, cinematic, a medium-wide shot frames ...")."""
    return (phrase or "").strip().rstrip(".,")


def render_summary(plan: Plan) -> str:
    """The [task type] prefix is templated from roles; only the sentence after it is prose.

    A prose stage that could write the prefix could invent a relationship the pack does not
    contain ('video editing' with no source video), so it never gets the chance.
    """
    prefix = "[" + " + ".join(plan.task_types) + "]"
    body = (plan.summary or "").strip()
    if body.startswith("["):                      # strip a prefix the model added anyway
        body = body.split("]", 1)[-1].strip()
    if not body:
        subs = ", ".join(s.label for s in plan.subjects) or "the described scene"
        body = f"The target video shows {subs}."
    edit = next((m.label for m in plan.manifest if m.role is Role.EDIT_SOURCE), None)
    if edit:
        required = f"The target video is an edited version of {edit}."
        # `startswith(required)` matched only the full-stop form, so a body already opening "The
        # target video is an edited version of <Video 1>, where the vehicle changes ..." had the
        # sentence prepended anyway and shipped it twice. The mandated clause is what matters and it
        # is allowed to continue; ref-en.txt 3 prints the stop, and a comma is the same sentence
        # carrying on.
        if not re.match(rf"The target video is an edited version of {re.escape(edit)}\s*[.,]", body):
            body = f"{required} {body}"
    return f"{prefix} {body}"


def dialogue_markup(line: DialogueLine, speaker: SpeakerPlan | None) -> str:
    """The user's words, byte-for-byte, inside the spec's markup."""
    sid = speaker.sid if speaker else "(S1)"
    subj = f"{speaker.subject} " if speaker and speaker.subject else ""
    desc = ""
    if speaker and speaker.descriptor:
        desc = f"{speaker.descriptor} "
    if speaker and speaker.voice_ref:
        voice = f", using the voice timbre referenced from {speaker.voice_ref},"
    else:
        voice = ""
    body = f"<d>[{line.language}] {line.text}</d>"
    if line.voiceover:
        return (f"{desc}{subj}{sid} says in an off-screen voiceover: {body} "
                "while their lips remain completely closed.")
    return f"{desc}{subj}{sid}{voice} says: {body}"


def _fill_shot_body(shot, plan: Plan, opts: ProfileOptions, notes: list[str]) -> str:
    """Pure: the shot is never mutated. Re-rendering the same plan must be byte-identical, so the
    rendered bodies travel out in RenderResult.shot_bodies for downstream checks to read."""
    body = (shot.body or "").strip()
    cam = shot.camera.phrase(opts.camera_style) if shot.camera else ""

    # Trimmed BEFORE any substitution. Running it afterwards deleted the canonical camera
    # sentence the template had just guaranteed, because the model had placed {{CAM}} late in a
    # long body and the tail went with the trim. Cutting while the placeholders are still tokens
    # means a lost {{CAM}} falls into the "token missing, appended" path and self-heals.
    # The template will ADD words after this point -- the canonical camera sentence and each
    # dialogue line -- so the ceiling budgets for them. Otherwise the final body exceeds the
    # number the trim was enforcing, which is the same "the budget does not mean anything" problem
    # in a smaller coat.
    added = (len(cam.split()) if cam else 0) + sum(
        len(d.text.split()) + 8 for d in shot.dialogue)
    ceiling = max(int(shot.word_target * 0.9), int(shot.word_target * 1.5) - added)
    words = body.split()
    if len(words) > ceiling:
        sentences = re.split(r"(?<=[.!?])\s+", body)
        kept, n = [], 0
        for sent in sentences:
            c = len(sent.split())
            if kept and n + c > ceiling:
                break
            kept.append(sent)
            n += c
        if kept and n >= shot.word_target * 0.6:
            notes.append(f"shot {shot.n}: trimmed {len(words) - n} words over the "
                         f"{ceiling}-word ceiling at a sentence boundary")
            body = " ".join(kept)


    if CAM_TOKEN in body:
        body = body.replace(CAM_TOKEN, cam + "." if cam and not cam.endswith(".") else cam)
    elif cam:
        # The prose stage dropped the token. Append rather than lose the camera move, and say so.
        notes.append(f"shot {shot.n}: camera token missing from prose, appended")
        body = body.rstrip()
        if body and not body.endswith((".", "!", "?")):
            body += "."
        body = f"{body} {cam}." if body else f"{cam}."

    # The prose stage is told to write nothing about camera movement, because the template owns
    # it. When it does anyway, the shot ends up with the canonical sentence AND a paraphrase of
    # it. Remove the paraphrase rather than shipping both: camera is a template decision, and a
    # second opinion about it is exactly what the placeholder exists to prevent.
    if cam:
        body, dropped = _strip_extra_camera_prose(body, cam)
        if dropped:
            notes.append(f"shot {shot.n}: removed {dropped} redundant camera sentence(s) "
                         "the prose stage added alongside the template's")

    # The label is the ONLY binding between this prose and the attached image. When the prose
    # stage describes a planned subject but omits its label, the shot silently stops referencing
    # the reference. That is structural, not stylistic, so the template repairs it rather than
    # reporting it: attach the label to the first mention of the subject's own descriptor.
    # REF2VA only. <Subject N> means something because subject_definitions defines it, and base
    # modes have no such section -- attaching one there creates a label that names nothing. This
    # is the third place that rule has had to be applied; it belongs to the label, not to a stage.
    for subj in (plan.subjects if plan.mode is Mode.REF2VA else []):
        if subj.label not in shot.subjects or subj.label in body:
            continue
        head = re.sub(r"^(?:a|an|the)\s+", "", subj.descriptor.strip(), flags=re.I)
        if not head:
            continue
        m = re.search(rf"\b((?:A|An|The|a|an|the)\s+)?{re.escape(head)}\b", body)
        if m:
            body = (body[:m.start()] + f"{subj.label}, {m.group(0).strip()}" + body[m.end():])
            notes.append(f"shot {shot.n}: attached {subj.label} to its first mention "
                         f"({head!r}); the prose described it without the label")

    used: set[int] = set()

    def _speaker_for(line):
        # The hint match is the binding; fallback only when no speaker_hint was given.
        # (append path previously forced speakers[0] for every token the prose dropped,
        # silently making BOTH lines share one speaker — control-plane overreach.)
        hint = line.speaker_hint or ""
        if hint:
            sp = next((s for s in plan.speakers if (s.descriptor or "") == hint), None)
            if sp:
                return sp
        return plan.speakers[0] if plan.speakers else None

    def _sub(m: re.Match) -> str:
        i = int(m.group(1))
        used.add(i)
        if 1 <= i <= len(shot.dialogue):
            return dialogue_markup(shot.dialogue[i - 1], _speaker_for(shot.dialogue[i - 1]))
        return ""

    body = DLG_TOKEN.sub(_sub, body)
    missing = [i for i in range(1, len(shot.dialogue) + 1) if i not in used]
    for i in missing:
        notes.append(f"shot {shot.n}: dialogue token {{{{D{i}}}}} missing from prose, appended")
        line = shot.dialogue[i - 1]
        sp = _speaker_for(line)
        body = body.rstrip()
        if body and not body.endswith((".", "!", "?")):
            body += "."
        body = f"{body} {dialogue_markup(line, sp)}".strip()

    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\.\s*\.+", ".", body)          # substitution can abut an existing stop
    body = re.sub(r"\s+([.,;:!?])", r"\1", body)
    if body and not body.endswith((".", "!", "?", '"', ">")):
        body += "."
    return body


def render_description(plan: Plan, opts: ProfileOptions, notes: list[str],
                       rendered: dict[int, str] | None = None) -> str:
    """Ref2VA puts shots on their own lines after a style sentence; base modes run inline
    with the style inside [Shot 1]. Both verified against MiniMax's published artifacts."""
    parts: list[str] = []
    for shot in plan.shots:
        body = _fill_shot_body(shot, plan, opts, notes)
        if rendered is not None:
            rendered[shot.n] = body
        if shot.n == 1:
            prefix = style_prefix(plan.style_phrase) if plan.mode.is_base else ""
            if prefix and body:
                parts.append(f"[Shot 1] {prefix}, {body[0].lower() + body[1:]}")
            elif prefix:
                parts.append(f"[Shot 1] {prefix}.")
            else:
                parts.append(f"[Shot 1] {body}")
        else:
            parts.append(f"[Shot {shot.n}] At {ms_to_timestamp(shot.start_ms)}, "
                         f"{_continue_case(body)}")

    if plan.mode is Mode.REF2VA:
        opening = style_sentence(plan.style_phrase)
        return "\n".join(([opening] if opening else []) + parts)
    return " ".join(parts)


CAMERA_SENTENCE = re.compile(r"(?:^|(?<=[.!?])\s*)The camera\b[^.!?]*[.!?]")


def _strip_extra_camera_prose(body: str, canonical: str) -> tuple[str, int]:
    keep = canonical.rstrip(".")
    dropped = 0

    def repl(m: re.Match) -> str:
        nonlocal dropped
        if keep and keep in m.group(0):
            return m.group(0)
        dropped += 1
        return " "

    out = CAMERA_SENTENCE.sub(repl, body)
    # The prose stage sometimes treats {{CAM}} as the SUBJECT of its sentence and carries on
    # through it ("{{CAM}} pushes steadily forward, closing the distance"). After substitution
    # that leaves a lowercase fragment stranded behind a full stop, which no camera-sentence
    # pattern matches because it does not start with "The camera".
    anchor = keep + "."
    idx = out.find(anchor)
    if idx >= 0:
        tail = out[idx + len(anchor):]
        m = re.match(r"\s+([a-z][^.!?]*[.!?])", tail)
        if m and re.match(r"(?:pushes|pulls|zooms|pans|trucks|tilts|rises|lowers|arcs|tracks|"
                          r"holds|shakes|rolls|moves|glides|drifts|closes)\b", m.group(1)):
            out = out[:idx + len(anchor)] + tail[m.end():]
            dropped += 1
    return re.sub(r"\s{2,}", " ", out).strip(), dropped


DETERMINERS = {"A", "An", "The"}


def _continue_case(body: str) -> str:
    """After 'At MM:SS.mmm,' the sentence continues, so the first word is lower-case -- as it is
    in every published example. Only lowered when provably safe: a determiner, or a word that
    already appears in lower case elsewhere in the same body (so proper nouns are left alone)."""
    if not body:
        return body
    first = re.match(r"[\w'-]+", body)
    if not first:
        return body
    w = first.group(0)
    if not w[:1].isupper():
        return body
    if w in DETERMINERS or re.search(rf"(?<![.!?]\s)\b{re.escape(w.lower())}\b", body[len(w):]):
        return body[0].lower() + body[1:]
    return body


def render_ir(plan: Plan, opts: ProfileOptions | None = None) -> RenderResult:
    opts = opts or ProfileOptions()
    notes: list[str] = []
    rendered: dict[int, str] = {}
    sep = opts.field_separator

    desc = render_description(plan, opts, notes, rendered)
    soundscape = sentences(plan.ambient_sound) or "N/A"
    music = plan.music.strip() or "N/A"

    if plan.mode is Mode.REF2VA:
        sections = {
            "subject_definitions": render_subject_definitions(plan),
            "summary": render_summary(plan),
            "retention_analysis": render_retention(plan),
            "detailed_description": desc,
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
        }
        # Ref2VA: section name on its own line, content following.
        prompt = sep.join(f"{k}:\n{v}" for k, v in sections.items())
    else:
        sections = {
            "integrated_multimodal_description": desc,
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
        }
        # Base modes: field name and content on the same line.
        body = sep.join(f"{k}: {v}" for k, v in sections.items())
        instr = instruction_line(plan, opts)
        prompt = f"{instr}\n\n{body}" if instr else body

    return RenderResult(prompt=prompt.strip() + "\n", sections=sections, notes=notes,
                        shot_bodies=rendered)
