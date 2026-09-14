"""What crosses the boundary between the compiler and anything that drives it, published.

The compiler and the ComfyUI node pack ship to two different audiences and are becoming two
repositories. A handful of facts have to be true in both of them at once, and until now the only
thing holding each one true was a test that opened the other half's source file and read it as text.
Those tests cannot exist once the halves live apart, and two of them were already guarding the wrong
hop -- see `tests/test_contract.py` for the one that passed while the field it guarded was being
dropped in transit.

So the compiler states the contract instead of a test inferring it. One module, built from the
authorities rather than restating them, published three ways:

    import          `from h3ir.contract import contract` -- for a caller in the same Python
    HTTP            `GET /v1/contract` -- for a caller on another machine
    command         `h3ir contract` -- for a human, and for whoever regenerates a copy

**Nothing here is a second source of truth.** Every section is read off the thing that already owns
it: `Role` for the roles, `director.DIRECTORS` for the profiles, `grid` for the ceilings, `shots`
for the shot cap. The one exception is the wire field names, and it is deliberate: those live on
pydantic models in `service.py`, this module may not import fastapi, and
`test_contract_matches_the_wire_models` pins the two together from the compiler's own side where
both are present. A literal with a test on it is honest; a literal without one is the drift this
module exists to end.

**`CONTRACT_VERSION` is its own number and it is not the package version.** A package version moves
for reasons that touch nothing a client can see. This moves only when one of the sections below
changes, and `tests/test_contract.py` holds the digest of every section against it, so changing a
director's prose without saying the contract changed is a red test rather than a discovery someone
makes later.

**What a client does about a difference is the client's business, not this module's.** Nothing here
refuses anything. It publishes facts and the digests that make comparing them cheap; the node pack
decides which differences stop a queue and which are worth a line in a report, because only the
caller knows what its own graph is about to send. See `contract.py` in the node pack.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import director as D
from .grid import (FPS, MAX_REF_AUDIOS, MAX_REF_IMAGES, MAX_REF_VIDEO_SOUNDTRACKS, MAX_REF_VIDEOS,
                   TRAINED_MAX_FRAMES, TRAINED_MIN_FRAMES)
from .models import Role
from .shots import PINNED_SHOTS_MAX

# Bump when any section below changes. `tests/test_contract.py` holds a digest per section and fails
# until this number moves with them, which is what stops the version being a number nobody maintains.
#
# There is no compatibility range published here on purpose. A client cannot usefully be told "1 and
# 2 are interchangeable" -- what it needs to know is whether the particular field, role or limit IT
# is about to use exists on the other side, and that is answered by comparing the sections, not the
# number. The number is what a message quotes so two people can say which contract they each have.
CONTRACT_VERSION = 3


# --------------------------------------------------------------------------- the wire

# WHICH SURFACE THESE DESCRIBE, because getting it wrong builds the wrong thing.
#
# `ASSET_FIELDS`, `BRIEF_FIELDS` and `DIALOGUE_FIELDS` are the keys of a REQUEST -- the JSON a
# caller posts to `POST /v1/briefs`. They are not the fields of `models.Brief` and `models.AssetRef`,
# which are the compiler's own dataclasses and carry a different set: `role_stated`, `px`,
# `composition`, `canvas` and more, none of which a caller may state.
#
# That distinction matters now that a client can run the compiler in the same Python. Measured:
# every compiler module imports with fastapi and pydantic absent EXCEPT `service.py`, which is
# where `_to_brief` -- the only conversion from a request into a `Brief` -- lives, along with the
# refusals it raises. So an in-process caller either drags fastapi into its host to reuse that
# conversion, or builds the dataclasses itself and takes on `role_stated` and the pairing rules.
# A caller that builds dataclasses gets its field names checked by Python at call time, which is
# loud and needs no contract; what it does NOT get checked is everything below this block, and
# that is where the contract earns its place on both paths.
ROLE_OF_THE_FIELD_LISTS = (
    "the keys of a POST /v1/briefs request. Not the fields of h3ir.models.Brief and "
    "h3ir.models.AssetRef, which carry more and are the compiler's own, not a caller's.")

# Every key one attachment may carry in a request, which is `service.AssetIn`'s field list.
#
# Written out rather than read off the model because this module has to import with fastapi and
# pydantic absent -- the node pack calls the compiler in-process and ComfyUI's Python is not the
# compiler's. `test_contract_matches_the_wire_models` holds the two together from the compiler's own
# side, where both are installed.
#
# This is the list whose drift is the quietest failure in the whole system: pydantic used to drop an
# unknown key in silence, so a picture arrived saying nothing about who it replaces, and that
# compiles, validates and renders the wrong swap. `AssetIn` forbids extras now and names the key it
# did not know, and this list is how a client can tell before it spends the request.
ASSET_FIELDS: tuple[str, ...] = (
    "path", "sha256", "url", "note", "replaces", "kind", "role", "sizing", "seconds", "frames",
    "paired_video_path", "paired_video_sha256", "provenance",
)

# Every top-level key of a request, which is `service.BriefIn`'s field list. Same reasoning, same
# test.
BRIEF_FIELDS: tuple[str, ...] = (
    "intent", "assets", "seconds", "aspect", "megapixels", "dialogue", "onscreen_text", "shots",
    "loras", "silent", "constraints", "creativity", "director", "director_profile", "effort",
    "seed", "transcripts", "answer", "compile_mode", "mode",
)

# Every key of one dialogue line, which is `service.DialogueIn`'s field list.
DIALOGUE_FIELDS: tuple[str, ...] = ("text", "language", "speaker", "voiceover")


# Every refusal a client can be handed: the status it arrives on, and which call can produce it. A
# client branches on these to say something useful instead of printing a status code, so a code
# raised over here with no branch over there is a caller that shrugs at a failure it could have
# explained.
#
# `on` matters as much as `status`. The node pack talks to two routes and their messages are not
# interchangeable: a failed brief is explained in terms of the graph, and a failed upload has to
# name the tray slot the file came from, because to the service an uploaded file IS its content
# hash and a message naming a hash tells somebody with nine references nothing. One code is raised
# on both, so this is a list rather than a single name.
#
# Declared rather than scanned, for the reason the field lists above are: a scan of source text is
# not something a client on another machine can run. `test_contract.py` holds it against the source
# from the compiler's own side, where the source is.
#
# Twelve of these are raised by `compile.py` as a `BriefRefused` and re-raised by the brief route,
# so a scan of `service.py` alone cannot see any of them. Only `over-capacity` was ever noticed,
# because it is the one raised through a subclass and the scan looking for them looked for
# `super().__init__`. The other eleven -- every refusal about a contradictory request, including all
# four about who a picture replaces -- were invisible to the check that was meant to prove every
# refusal reaches a client with a sentence attached.
BRIEFS, ASSETS = "briefs", "assets"
ERROR_CODES: dict[str, dict[str, Any]] = {
    "analysis-tool-missing": {"status": 503, "on": [BRIEFS]},
    "aspect-invalid": {"status": 422, "on": [BRIEFS]},
    "asset-digest-mismatch": {"status": 422, "on": [ASSETS]},
    "asset-missing": {"status": 422, "on": [BRIEFS]},
    "asset-name-not-a-digest": {"status": 422, "on": [ASSETS, BRIEFS]},
    "asset-no-path": {"status": 422, "on": [BRIEFS]},
    "asset-not-uploaded": {"status": 422, "on": [BRIEFS]},
    "asset-paths-disabled": {"status": 422, "on": [BRIEFS]},
    "asset-too-large": {"status": 413, "on": [ASSETS]},
    "asset-two-sources": {"status": 422, "on": [BRIEFS]},
    "asset-unreadable": {"status": 422, "on": [BRIEFS]},
    "change-empty": {"status": 422, "on": [BRIEFS]},
    "director-profile-invalid": {"status": 422, "on": [BRIEFS]},
    "duration-invalid": {"status": 422, "on": [BRIEFS]},
    "intent-empty": {"status": 422, "on": [BRIEFS]},
    "llm-error": {"status": 502, "on": [BRIEFS]},
    "llm-unavailable": {"status": 503, "on": [BRIEFS]},
    "malformed-request": {"status": 422, "on": [BRIEFS]},
    "over-capacity": {"status": 422, "on": [BRIEFS]},
    "replacement-subject-undefined": {"status": 422, "on": [BRIEFS]},
    "replacement-target-ambiguous": {"status": 422, "on": [BRIEFS]},
    "replacement-target-unnamed": {"status": 422, "on": [BRIEFS]},
    "replaces-without-the-role": {"status": 422, "on": [BRIEFS]},
    "shots-do-not-fit": {"status": 422, "on": [BRIEFS]},
    "shots-invalid": {"status": 422, "on": [BRIEFS]},
    "swap-without-edit-source": {"status": 422, "on": [BRIEFS]},
    "unknown-brief": {"status": 404, "on": [BRIEFS]},
    "unknown-field": {"status": 422, "on": [BRIEFS]},
    "unknown-role": {"status": 422, "on": [BRIEFS]},
    "upload-store-full": {"status": 507, "on": [ASSETS]},
}


# --------------------------------------------------------------------------- the vocabularies

# Which roles make sense on which kind of file. This used to live in `service.py` and be read only
# to build an error message, which is how `structure` came to be in the enum and missing from the
# list for as long as it was: the effect was a caller told a legal role does not exist. It is a
# published vocabulary now and the message is one of its readers rather than its owner.
#
# The keys are `AssetKind`'s own words. A surface may call a picture a picture; the wire says image.
ROLES_BY_KIND: dict[str, tuple[Role, ...]] = {
    "image": (Role.FRAME_ANCHOR_FIRST, Role.FRAME_ANCHOR_LAST, Role.SUBJECT, Role.ENVIRONMENT,
              Role.STYLE, Role.STORYBOARD, Role.PLACED_SUBJECT, Role.REPLACEMENT_SUBJECT),
    "video": (Role.EDIT_SOURCE, Role.CONTINUATION_SOURCE, Role.SUBJECT, Role.ENVIRONMENT,
              Role.STYLE, Role.STORYBOARD, Role.STRUCTURE),
    "audio": (Role.VOICE_TIMBRE, Role.BGM, Role.MUSIC_STYLE, Role.BEAT_REFERENCE, Role.SFX),
}

# The dial, the shapes and the languages a request may name. All three were literals inside
# `service.capabilities()` and inside the node pack at once; they are here so that is one statement.
# The languages are a published list rather than a limit -- the compiler writes whatever language it
# is given into the `[tag]` H3 reads -- and the field's own tooltip says so.
CREATIVITY: tuple[str, ...] = ("restrained", "balanced", "bold", "extreme")
EFFORT: tuple[str, ...] = ("fast", "standard", "max")
SIZING: tuple[str, ...] = ("match", "max")
ASPECTS: tuple[str, ...] = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
DIALOGUE_LANGUAGES: tuple[str, ...] = (
    "Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean",
    "Portuguese", "Russian", "Spanish",
)


def _limits() -> dict[str, Any]:
    """Every number a surface restates so it can refuse or offer something before a request.

    Each one is read off the module that enforces it. A surface holding a stale copy of one of these
    does not render wrongly -- it refuses something legal, or offers something the compiler then
    turns away -- so these are worth reporting and never worth stopping a queue over.
    """
    return {
        "director_notes_max_chars": D.MAX_NOTES_CHARS,
        "max_pinned_shots": PINNED_SHOTS_MAX,
        "fps": FPS,
        "trained_frames": [TRAINED_MIN_FRAMES, TRAINED_MAX_FRAMES],
        "max_assets": {
            "images": MAX_REF_IMAGES,
            "videos": MAX_REF_VIDEOS,
            "audios": MAX_REF_AUDIOS,
            "video_soundtracks": MAX_REF_VIDEO_SOUNDTRACKS,
        },
        "aspects": list(ASPECTS),
        "creativity": list(CREATIVITY),
        "effort": list(EFFORT),
        "sizing": list(SIZING),
        "dialogue_languages": list(DIALOGUE_LANGUAGES),
    }


# --------------------------------------------------------------------------- assembling it

# The sections a client compares one at a time. Named here rather than inferred from the dict's keys
# so that adding a section is a decision somebody made, and so `digests()` and every consumer read
# the same list in the same order.
SECTIONS: tuple[str, ...] = ("asset_fields", "brief_fields", "dialogue_fields", "error_codes",
                             "roles", "directors", "camera_moves", "limits")


def _sections() -> dict[str, Any]:
    return {
        "asset_fields": list(ASSET_FIELDS),
        "brief_fields": list(BRIEF_FIELDS),
        "dialogue_fields": list(DIALOGUE_FIELDS),
        "error_codes": {code: dict(spec) for code, spec in ERROR_CODES.items()},
        "roles": {kind: [r.value for r in roles] for kind, roles in ROLES_BY_KIND.items()},
        "directors": [D.to_mapping(d) for d in D.DIRECTORS],
        "camera_moves": list(D.CAMERA_MOVES),
        "limits": _limits(),
    }


def digest(value: Any) -> str:
    """The name a section has while it is unchanged.

    Canonical JSON so that a dict written in another order hashes the same, and sixteen hex
    characters because this is a name to quote in a sentence, not a signature. `ensure_ascii` is off
    so the profiles hash as the characters they are rather than as their escapes -- an em dash is a
    character in that prose and a change to it is a change to what the writer is sent.
    """
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def digests() -> dict[str, str]:
    """One name per section, so a client can say WHICH part differs instead of that something does."""
    sections = _sections()
    return {name: digest(sections[name]) for name in SECTIONS}


def snapshot() -> dict[str, Any]:
    """The contract as a REPRODUCIBLE document: the same bytes from any install of this build.

    Nothing in here describes a particular installation, which is what makes it safe to write to
    disk and diff. `contract()` below is this plus the one field that does.
    """
    out: dict[str, Any] = {"contract_version": CONTRACT_VERSION,
                           "field_lists_describe": ROLE_OF_THE_FIELD_LISTS}
    out.update(_sections())
    out["digests"] = digests()
    return out


def contract() -> dict[str, Any]:
    """What a live service answers with: the snapshot, and which build is answering.

    `package_version` is deliberately absent from `snapshot()` and present here. A client asking a
    running service wants to be able to say "the one at this address is 0.2.0"; a copy written into
    another repository has no running service behind it, and baking one install's metadata into a
    generated file would make regenerating it produce different bytes on different machines. That is
    not hypothetical: this checkout's own metadata says 0.1.0 while `pyproject.toml` says 0.2.0,
    because the editable install predates the bump.
    """
    out = snapshot()
    out["package_version"] = _package_version()
    return out


def _package_version() -> str:
    """What this build calls itself, for a message that has to name a version to install.

    Informational only, and never compared: two builds at one package version always agree about the
    contract, and two at different versions usually do. `contract_version` is the field that answers
    whether they agree.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                    # pragma: no cover - 3.7 and older only
        return ""
    try:
        return version("open-h3-ir")
    except PackageNotFoundError:                           # running from a checkout, not installed
        return ""


# --------------------------------------------------------------------------- publishing a copy

def as_json() -> str:
    """The contract as a file somebody can open, diff and read.

    Indented and newline-terminated because this is written to disk in the node pack's repository
    and read by whoever is working out why their pack and their compiler disagree. A minified blob
    would make every regeneration a one-line diff nobody can review.

    The SNAPSHOT rather than the live contract, so that two people regenerating it from the same
    build get the same file. See `contract()` for the one field that differs.
    """
    return json.dumps(snapshot(), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def as_js() -> str:
    """The three things a BROWSER needs, as an ES module the node pack ships.

    The director panel has to show the seven profiles, the twenty camera moves and the length cap
    with no compiler running anywhere: a text box that needs a service before it can show you a
    paragraph is a text box that is empty exactly when somebody is trying to write in it. So the
    pack carries a copy, and that copy is GENERATED from here rather than typed, because eleven
    thousand characters of prose maintained by hand in two languages is drift with a schedule.

    Only these three. The wire keys, the roles and the limits are the pack's Python's business and
    reach it through the pack's `contract.json`; the browser draws a picker and nothing else.

    `JSON.stringify`-shaped values rather than hand-quoted strings, so no escape in the prose has to
    be thought about twice.
    """
    c = snapshot()
    parts = [
        "/* GENERATED by `h3ir contract --js`. Do not edit.\n"
        " *\n"
        " * The seven directions, the twenty camera moves H3 has names for, and the cap the\n"
        " * compiler refuses a longer direction at. The compiler owns all three; this file is the\n"
        " * copy the panel draws from so the picker works with no service running.\n"
        " *\n"
        " * To refresh: run the command above with the matching open-h3-ir installed, and write its\n"
        " * output over this file. `tests/test_contract_drift.py` fails while it is stale.\n"
        " */\n",
        f"export const CONTRACT_VERSION = {c['contract_version']};",
        f"export const DIRECTORS_DIGEST = {json.dumps(c['digests']['directors'])};",
        f"export const MAX_NOTES = {c['limits']['director_notes_max_chars']};",
        "",
        "export const CAMERA_MOVES = " + _js_list(c["camera_moves"]) + ";",
        "",
        "export const DIRECTORS = [",
    ]
    for d in c["directors"]:
        parts.append("  {")
        parts.append(f"    id: {json.dumps(d['id'], ensure_ascii=False)},")
        parts.append(f"    name: {json.dumps(d['name'], ensure_ascii=False)},")
        parts.append(f"    notes: {json.dumps(d['notes'], ensure_ascii=False)},")
        parts.append("  },")
    parts.append("];")
    return "\n".join(parts) + "\n"


def _js_list(values: list[str]) -> str:
    """A JavaScript array of strings, wrapped so the file stays readable at a normal width."""
    lines: list[str] = ["["]
    row = " "
    for v in values:
        item = " " + json.dumps(v, ensure_ascii=False) + ","
        if len(row) + len(item) > 96:
            lines.append(row)
            row = " "
        row += item
    lines.append(row)
    lines.append("]")
    return "\n".join(lines)
