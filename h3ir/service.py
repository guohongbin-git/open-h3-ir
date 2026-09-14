"""The HTTP surface. The API is the product; the UI is one of its clients.

The rule that keeps both callers honest: there is no quality-bearing field only a UI could
set, and no path that skips the validator. A one-sentence brief from an agent and a fully
specified brief from a UI enter the same pipeline at the same point.

Three response layers, so nobody has to see "ref2va":
  presentation  plain language for an end user. No mode names, no frame counts, no nodes.
  plan          the creative decisions, reviewable and refinable.
  ir            the full document, for whoever debugs or reproduces it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import contract as C
from . import uploads
from .analyse import AssetAnalysisError, ToolMissing, sha256_file
from .backend import Backend, BackendError, BackendUnavailable
from .compile import BriefRefused, compile_brief, refine
from .config import get_config
from .mode import needs_clarification
from .models import AssetKind, AssetRef, Brief, DialogueLine, IRDocument, Mode, Role
from .plan import ProfileOptions

log = logging.getLogger("h3ir.service")

app = FastAPI(title="h3ir", version="1",
              description="Compiles a plain-language brief into a validated MiniMax H3 "
                          "Context-IR plus the exact asset wiring it is true for.")

_STORE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- wire models

class AssetIn(BaseModel):
    """One attachment, and exactly one statement of where its bytes are.

    **Unknown keys are refused, not dropped.** pydantic ignores an extra key by default, and that
    default is the quietest failure this service can have: a client sending `replaces` to a build
    that does not know the word gets a perfectly valid brief back with the swap bound to whoever the
    analyser happened to find, and nothing anywhere says so. Forbidding extras turns that into a 422
    that names the key, which a newer client can read and act on. `h3ir.contract.ASSET_FIELDS` is
    the same list, published so a client can check before it spends the request.

    `path` is the fast way and needs one filesystem: nothing is copied, and the bytes H3 is
    conditioned on are the bytes the caller already had. `sha256` is the way in for a caller on
    another machine, naming an attachment it has already sent to `PUT /v1/assets/{sha256}`.

    Both at once is refused rather than one winning. They can disagree, and a request that says two
    things about which file it means renders something plausible about the wrong one.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(None, description="Readable path on the service host")
    sha256: str | None = Field(
        None, description="An attachment already uploaded to this service, named by the sha256 of "
                          "its own bytes. Use instead of `path` when the caller and the service do "
                          "not share a filesystem. Upload with PUT /v1/assets/{sha256}.")
    url: str | None = None
    note: str | None = Field(None, description="Free text hint, e.g. 'the man'. Never required.")
    replaces: str | None = Field(
        None, description="On a `replacement_subject` picture only: who in the edited clip this "
                          "one takes over from, in your own words, e.g. 'the man in the plaid "
                          "shirt'. Needed when the clip holds more than one figure of the same "
                          "kind, and when the figure is not visible in the frames this service "
                          "samples. Refused on any other role rather than dropped.")
    kind: Literal["image", "video", "audio"] = "image"
    role: str | None = Field(None, description="Optional. Inferred when omitted.")
    sizing: Literal["match", "max"] = "match"
    seconds: float | None = None
    frames: int | None = None
    paired_video_path: str | None = None
    paired_video_sha256: str | None = Field(
        None, description="The uploaded video this soundtrack belongs to, named by its sha256. The "
                          "upload counterpart of `paired_video_path`.")
    provenance: dict[str, Any] | None = None


class DialogueIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    language: str = "English"
    speaker: str | None = None
    voiceover: bool = False


class BriefIn(BaseModel):
    """The minimum viable request is `intent`. Everything else has a good default.

    Unknown keys are refused for the reason `AssetIn` gives: a key this build does not know is a
    caller asking for something it will not get, and a silent drop is the one answer nobody can act
    on. `h3ir.contract.BRIEF_FIELDS` publishes the list.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str
    assets: list[AssetIn] = []
    seconds: float = 5.0
    aspect: str = "16:9"
    megapixels: float | None = Field(
        None, ge=0.25, le=2.5,
        description="Canvas size as a pixel-area ask, like a resolution picker's 1.5. Omitted "
                    "means H3's native geometry, 768 on the short edge, which is what the model "
                    "was trained at. Larger renders work and cost time and VRAM in proportion; "
                    "the trained sweet spot is the default.")
    dialogue: list[DialogueIn] = []
    onscreen_text: list[str] = []
    shots: int | None = None
    loras: list[dict[str, Any]] = []
    silent: bool = False
    constraints: list[str] = []
    creativity: Literal["restrained", "balanced", "bold", "extreme"] = Field(
        "balanced",
        description="How much the writer can add beyond what the request states. "
                    "restrained = the request and nothing else. balanced = the request, shaped, "
                    "plus music if the video wants it. bold = can introduce a spoken line, "
                    "music, on-screen text or a beat the request never mentioned. An explicit "
                    "prohibition in the request (\"no dialogue\") overrides every setting.")
    director: str = Field(
        "",
        description="One of the profiles that ship with this service, by id: whose taste fills what "
                    "the request and the references leave open: the camera, the framing, the light "
                    "and colour, what the frame attends to, how bodies and delivery are written, "
                    "and what the room and any music are made of. It never decides how many shots "
                    "there are or where they cut, and anything the request states explicitly "
                    "outranks it. GET /v1/directors for the list and the prose of each. Empty means "
                    "no direction, which is what every brief written before this existed compiles "
                    "at.")
    director_profile: dict[str, Any] | None = Field(
        None,
        description="Direction in your own words, instead of a shipped id: {\"name\": ..., "
                    "\"notes\": ...}. `notes` is prose, how the video should be shot, and it is "
                    "handed to the writer as taste, not as rules: it is placed under the statement "
                    "of which attributes the request and the references already govern, and under "
                    "the creativity setting's prohibitions. `name` reaches the report and never the "
                    "writer, because naming a director invites imitation of particular films "
                    "instead of a way of working. Fetch a shipped profile and edit it to start from "
                    "one. Say habits, not shots.")
    effort: Literal["fast", "standard", "max"] = "standard"
    seed: int | None = 7
    compile_mode: Literal["draft_only", "enriched_required"] = Field(
        "enriched_required",
        description="draft_only = the deterministic draft is the deliverable (no prose model "
                    "call; zero prose variance; the benchmark path). enriched_required = the "
                    "prose pass runs and must succeed; a fallback to draft is reported, never "
                    "counted as success.")
    mode: Literal["ref2va", "fl2va", "t2va"] | None = Field(
        None,
        description="Caller override of the task mode. Locks the mode deterministically "
                    "(caller-override path, no model involved) — required for benchmark "
                    "compiles where the mode must never be inferred.")
    transcripts: dict[str, str] = Field(
        default_factory=dict,
        description="sha256 -> transcript, for attached audio. This service NEVER transcribes: "
                    "nothing here can hear, and a model asked about a waveform invents a plausible "
                    "answer instead of abstaining. Run a real recogniser — Whisper, or any "
                    "speech-to-text service — and pass the result. A transcript gives the "
                    "WORDS only — "
                    "timbre, delivery and tempo still have to be stated in the asset's note.")
    answer: dict[str, str] | None = Field(
        None, description="Optional answer to a previous clarification, e.g. "
                          '{"anchor_or_reference": "opening_frame"}')


class RefineIn(BaseModel):
    change: str


# --------------------------------------------------------------------------- conversion

_ROLE_BY_NAME = {r.value: r for r in Role}

# Which roles make sense for which kind. It used to live here and be read only to build an error
# message, which is how `structure` came to be in the enum and missing from the video line for as
# long as it was: the effect was a caller told a legal role does not exist. It is a published
# vocabulary in `contract.py` now, and this file is one of its readers rather than its owner.


def _role_fits(role: Role, kind: str) -> bool:
    return role in C.ROLES_BY_KIND.get(kind, ())


def _require_uploaded(b: BriefIn) -> None:
    """Refuse a request naming uploads this service does not hold, listing every one of them.

    In one pass over the whole request rather than per asset, and that is the point: a caller with
    nine references needs one answer telling it which nine to send, not nine round trips each
    naming the next file. `missing` in the detail is what a client uploads and retries.
    """
    wanted: list[str] = []
    for a in b.assets:
        for named in (a.sha256, a.paired_video_sha256):
            if not named:
                continue
            if not uploads.is_digest(named):
                raise HTTPException(422, detail={
                    "code": "asset-name-not-a-digest",
                    "message": f"{named[:80]!r} is not a sha256, so it cannot name an uploaded "
                               "attachment. An upload is named by the hash of its own bytes, as 64 "
                               "lowercase hexadecimal characters."})
            wanted.append(named)
    absent = uploads.missing(wanted)
    if absent:
        raise HTTPException(422, detail={
            "code": "asset-not-uploaded",
            "message": ("this service does not hold "
                        + (f"the attachment {absent[0]}" if len(absent) == 1
                           else f"{len(absent)} of the attachments this request names")
                        + ". Send the bytes to PUT /v1/assets/{sha256} and ask again. An upload is "
                          "kept for a while and then dropped, so one that worked an hour ago can "
                          "need sending a second time."),
            "missing": absent})


def _source_of(a: AssetIn) -> tuple[Path, str]:
    """Where this asset's bytes are and what they hash to, from whichever way it was attached."""
    if a.path and a.sha256:
        raise HTTPException(422, detail={
            "code": "asset-two-sources",
            "message": f"this asset gives both a path ({a.path}) and an uploaded sha256 "
                       f"({a.sha256}). They can name different files, so pick one: a path for a "
                       "service that can open your filesystem, an upload for one that cannot."})
    if a.sha256:
        p = uploads.stored(a.sha256)
        if p is None:
            # `_require_uploaded` already looked, so reaching here means the asset was evicted in
            # between. Reported the same way, because the caller's move is the same: send it again.
            raise HTTPException(422, detail={
                "code": "asset-not-uploaded",
                "message": f"the upload {a.sha256} was dropped from this service between one check "
                           "and the next. Send the bytes to PUT /v1/assets/{sha256} and ask again.",
                "missing": [a.sha256]})
        # Not re-hashed. The store's own name IS this digest: the bytes were streamed through
        # hashlib on the way in and the file was placed under what that produced, so re-reading a
        # 200 MB clip on every compile would confirm something already proved once.
        uploads.touch(p)
        return p, a.sha256
    if not a.path:
        raise HTTPException(422, detail={
            "code": "asset-no-path",
            "message": "each asset needs either a `path` this service can open or the `sha256` of "
                       "an attachment already uploaded to it. URL fetching is not enabled on this "
                       "deployment."})
    if not get_config().assets.allow_paths:
        raise HTTPException(422, detail={
            "code": "asset-paths-disabled",
            "message": "this service does not open files from its own filesystem: it was started "
                       "with H3IR_ALLOW_ASSET_PATHS off. Upload the attachment instead, with PUT "
                       "/v1/assets/{sha256}, and name it by that sha256."})
    p = Path(a.path)
    if not p.exists():
        raise HTTPException(422, detail={"code": "asset-missing",
                                         "message": f"no such file: {a.path}"})
    return p, sha256_file(p)


def _to_brief(b: BriefIn) -> Brief:
    _require_uploaded(b)
    assets: list[AssetRef] = []
    for a in b.assets:
        p, sha = _source_of(a)
        role = _ROLE_BY_NAME.get(a.role or "", None)
        if role is None and (a.role or "").strip():
            # A role that does not resolve used to fall through to the kind default in silence, so
            # `frame_anchor` -- one word short of `frame_anchor_first` -- got a content reference and
            # the caller was never told. Anchor versus reference is the product's central
            # distinction, and this field is the only place a caller can state it.
            raise HTTPException(422, detail={
                "code": "unknown-role",
                "message": f"{a.role!r} is not a role. For "
                           f"{'an' if a.kind[0] in 'aeiou' else 'a'} {a.kind} the roles are: "
                           + ", ".join(r.value for r in Role if _role_fits(r, a.kind))
                           + ". Omit `role` to have it inferred from the request"})
        stated = role is not None
        if role is None:
            role = {"image": Role.SUBJECT, "video": Role.EDIT_SOURCE,
                    "audio": Role.BGM}[a.kind]
        paired = None
        if a.paired_video_sha256:
            # The upload half of the pointer below, and it needs no file at all: the digest IS the
            # identity the runtime pairs on, and `_require_uploaded` has already proved this service
            # holds those bytes.
            paired = a.paired_video_sha256
        elif a.paired_video_path:
            # Refused rather than ignored. A soundtrack whose pointer cannot be resolved gets
            # numbered as a standalone <Audio j>, while the runtime that receives it as
            # ref_video_audio_k emits its label immediately before the video's: two different
            # labels for one file, and a caller who asked for the pairing would never learn it did
            # not happen. `asset-missing` is also the one code callers retry a path translation on.
            if not get_config().assets.allow_paths:
                # The same gate as `path`, and it was missing here first: a pointer is a path, and
                # this one reaches `sha256_file`. A deployment that has turned filesystem reads off
                # would still have opened and hashed any file named here, which is both a read it
                # said it would not do and an oracle for guessed contents.
                raise HTTPException(422, detail={
                    "code": "asset-paths-disabled",
                    "message": "this service does not open files from its own filesystem, and "
                               "`paired_video_path` names one: it was started with "
                               "H3IR_ALLOW_ASSET_PATHS off. Upload the video and pair the "
                               "soundtrack to it with `paired_video_sha256` instead."})
            paired_path = Path(a.paired_video_path)
            if not paired_path.exists():
                raise HTTPException(422, detail={
                    "code": "asset-missing",
                    "message": f"no such file: {a.paired_video_path} (given as the video this "
                               "soundtrack is paired with)"})
            paired = sha256_file(paired_path)

    # An answered clarification becomes an explicit role, which is exactly how it would have
    # arrived had the caller known: the answer is data, not a special code path.
    if b.answer and b.answer.get("anchor_or_reference") == "opening_frame" and assets:
        assets[0].role = Role.FRAME_ANCHOR_FIRST

    return Brief(intent=b.intent, assets=assets, seconds=b.seconds, aspect=b.aspect,
                 megapixels=b.megapixels,
                 mode=Mode(b.mode) if b.mode else None,
                 dialogue=[DialogueLine(text=d.text, language=d.language,
                                        speaker_hint=d.speaker, voiceover=d.voiceover)
                           for d in b.dialogue],
                 onscreen_text=list(b.onscreen_text), shots=b.shots, loras=list(b.loras),
                 silent=b.silent, constraints=list(b.constraints), effort=b.effort,
                 creativity=b.creativity, director=b.director,
                 director_profile=b.director_profile)


def _plan_layer(doc: IRDocument) -> dict[str, Any]:
    """The creative decisions in plain language. No labels, no markers, no field names."""
    return {
        # The request is not judgeable without the ask beside it. A caller — or the person reading
        # over their shoulder — cannot tell whether a brief matched a plain request or overshot it
        # while looking only at the brief.
        "asked_for": doc.provenance.get("request"),
        "creativity": doc.provenance.get("creativity"),
        "director": doc.provenance.get("director"),
        "style": doc.plan.style_phrase,
        "shots": [{"n": s.n,
                   "from": round(s.start_ms / 1000, 2), "to": round(s.end_ms / 1000, 2),
                   "what_happens": s.beat,
                   "camera": (f"{s.camera.type}"
                              + (f", {s.camera.amplitude} amplitude" if s.camera and s.camera.amplitude else "")
                              + (f", {s.camera.speed}" if s.camera and s.camera.speed else ""))
                   if s.camera else "held",
                   "says": [d.text for d in s.dialogue],
                   "sound": s.sync_sound}
                  for s in doc.plan.shots],
        "background_sound": doc.plan.ambient_sound,
        "music": doc.plan.music,
        "people_who_speak": [{"as": s.sid, "is": s.descriptor or s.subject} for s in doc.plan.speakers],
        "references": [{"label": m.label, "role": m.role.value, "kind": m.kind.value}
                       for m in doc.plan.manifest],
    }


def _ir_layer(doc: IRDocument) -> dict[str, Any]:
    return {
        "ir_version": doc.ir_version, "profile": doc.profile, "mode": doc.mode.value,
        "prompt": doc.prompt, "prompt_tokens": doc.prompt_tokens,
        "sections": doc.sections,
        "target": {"nominal_seconds": doc.plan.target.nominal_seconds,
                   "frames": doc.plan.target.frames,
                   "effective_seconds": round(doc.plan.target.effective_seconds, 3),
                   "canvas": list(doc.plan.target.canvas)},
        "manifest": [asdict(m) for m in doc.plan.manifest],
        "mode_decision": asdict(doc.plan.mode_decision) if doc.plan.mode_decision else None,
        "loras": [asdict(l) for l in doc.plan.loras],
        "render_hash": doc.render_hash(),
        "provenance": doc.provenance,
        "diagnostics": [{"rule": f.rule, "severity": f.severity, "message": f.msg}
                        for f in doc.findings],
    }


def _envelope(brief_id: str, doc: IRDocument, brief: Brief) -> dict[str, Any]:
    clar = doc.provenance.get("clarification")
    return {
        "id": brief_id,
        "status": ("needs_input" if clar else ("degraded" if doc.fell_back else "ready")),
        "source": doc.source,
        "fallback_reason": doc.fallback_reason,
        "question": clar,
        "default_if_unanswered": (clar or {}).get("default_if_unanswered"),
        "presentation": doc.presentation(),
        "plan": _plan_layer(doc),
        "ir": _ir_layer(doc),
    }


def _remember(brief_id: str, brief: Brief, doc: IRDocument) -> None:
    with _LOCK:
        _STORE[brief_id] = {"brief": brief, "doc": doc, "at": time.time(),
                            "versions": _STORE.get(brief_id, {}).get("versions", 0) + 1}


# --------------------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict[str, Any]:
    with Backend(get_config()) as b:
        return {"ok": True, "llm": b.health(), "profile": get_config().profile}


@app.exception_handler(RequestValidationError)
async def unreadable_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A malformed request, answered in the same shape every other refusal here uses.

    FastAPI's own 422 body is a list of pydantic error objects, which is precise and unreadable, and
    every client in this repository reads `detail` as an object with a `code` and a `message`. So
    the list is turned into one sentence about the first thing wrong, and the raw list is kept
    beside it for whoever is debugging a client.

    **The `extra_forbidden` case is the one this handler exists for.** `AssetIn` and `BriefIn` used
    to drop a key they did not recognise, so a client one version ahead of this service sent
    `replaces` and got back a valid brief with the swap bound to whoever the analyser happened to
    find. Nothing said so. Now it is refused, and the refusal has to name the key and the fact that
    the two halves are at different versions -- a reader who only sees "extra inputs are not
    permitted" learns nothing about what to do.
    """
    problems = exc.errors()
    first = problems[0] if problems else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "the request"
    if first.get("type") == "extra_forbidden":
        detail = {
            "code": "unknown-field",
            "message": (
                f"this build does not know the field {where!r}. It was refused rather than "
                "ignored, because a field this service drops in silence is a request that comes "
                "back looking correct and describing something else. Whatever sent it is newer "
                "than this service: update open-h3-ir where the service runs (this one is "
                f"contract {C.CONTRACT_VERSION}, package "
                f"{C._package_version() or 'unknown'}), or stop sending the field. "
                "GET /v1/contract lists every field this build takes.")}
    else:
        detail = {
            "code": "malformed-request",
            "message": f"{where}: {first.get('msg', 'the request could not be read')}."}
    detail["problems"] = jsonable_encoder(problems)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/v1/contract")
def get_contract() -> dict[str, Any]:
    """Everything a client has to agree with this service about, and the digest of each part.

    Published so that a client checks itself against this build instead of a test reading across a
    folder that will not be there after the two halves are separate repositories. The node pack
    fetches this before it sends anything and compares it against what THIS graph is about to use,
    which is what lets an older service keep working for every brief that does not touch the part
    that moved.

    Cheap on purpose: no configuration is read, nothing is computed per request, and a client may
    call it on every queue.
    """
    return C.contract()


@app.get("/v1/directors")
def list_directors() -> dict[str, Any]:
    """Every profile a caller can name, with the prose of each.

    **The prose IS published, unlike `/v1/loras`' trigger strings.** A profile is not a secret and
    not a key: it is a paragraph an agent should be able to read, borrow a line from, and edit into
    its own. Withholding it would leave a caller choosing between seven names with nothing to tell
    them apart, which is the same thing the canvas panel refuses to do.
    """
    from . import director as D
    return {
        "governs": "whatever the request and the references leave open. An explicit instruction in "
                   "the request outranks a profile. Shot count and cut times are never a "
                   "profile's: pin `shots` and the count is yours, leave it unset and the writer "
                   "decides the edit.",
        "camera_moves": list(D.CAMERA_MOVES),
        "directors": [D.to_mapping(d) for d in D.DIRECTORS],
    }


@app.get("/v1/directors/{name}")
def get_director(name: str) -> dict[str, Any]:
    """One shipped profile, to read or to use as the starting point for your own."""
    from . import director as D
    d = D.BY_ID.get(str(name).strip().lower())
    if d is None:
        raise HTTPException(404, f"no director called {name!r}. The list is at GET /v1/directors.")
    return D.to_mapping(d)



@app.get("/v1/loras")
def list_loras() -> dict[str, Any]:
    """Everything an agent needs to choose a style. Deliberately no trigger strings: the
    compiler owns those, for the same reason it owns <Picture 1>."""
    from .lora import load_registry
    records, findings, revision = load_registry()
    return {"revision": revision,
            "loras": [r.public() for r in records.values()],
            "problems": [{"rule": f.rule, "severity": f.severity, "message": f.msg}
                         for f in findings]}


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    from .grid import (MAX_REF_AUDIOS, MAX_REF_IMAGES, MAX_REF_VIDEOS,
                       MAX_REF_VIDEO_SOUNDTRACKS, legal_frames)
    cfg = get_config()
    return {
        # Durations must be frame-ALIGNED (n % 17 == 5 at 24 fps). The band below is where the
        # model was trained; longer renders work, they just take longer, so it is reported as
        # guidance rather than as a limit.
        "durations_seconds": sorted({round(n / 24, 3) for n in legal_frames()}),
        "trained_band_seconds": [round(124 / 24, 3), round(362 / 24, 3)],
        "longer_durations": "supported if frame-aligned; slower, not rejected",
        "canvas": "any multiple of 32; 768 short edge is the default, larger renders fine",
        "aspects": list(C.ASPECTS),
        # The runtime's own socket maxima, and now enforced: over capacity is a 422 refusal naming
        # what to drop rather than a manifest that publishes a socket the graph does not have.
        # `total_files` used to say 12, which the runtime does not impose -- 9 images, 3 videos,
        # each video's soundtrack and 3 standalone audios all have sockets, so 18 attachments are
        # legal and a 12-file ceiling would have refused legal work.
        "max_assets": {"images": MAX_REF_IMAGES, "videos": MAX_REF_VIDEOS,
                       "audios": MAX_REF_AUDIOS,
                       "video_soundtracks": MAX_REF_VIDEO_SOUNDTRACKS,
                       "total_files": (MAX_REF_IMAGES + MAX_REF_VIDEOS + MAX_REF_AUDIOS
                                       + MAX_REF_VIDEO_SOUNDTRACKS)},
        "dialogue_languages": list(C.DIALOGUE_LANGUAGES),
        "output": {"short_edge": 768, "fps": C.contract()["limits"]["fps"],
                   "audio": "32 kHz stereo"},
        # Where the rest of what crosses a client boundary is published. Named from here because
        # `/v1/capabilities` is the route a client already knows to ask, and a client that has never
        # heard of the contract finds it by reading this.
        "contract": {"version": C.CONTRACT_VERSION, "endpoint": "GET /v1/contract"},
        # How attachments can reach this deployment, which is not the same on every one of them and
        # is not something a caller can find out by trying: a client that can read this refuses an
        # oversized file locally, in one sentence, instead of spending the transfer to be told.
        # `paths` false is the answer to "why did my path stop working", so it is published rather
        # than left to a 422.
        "assets": {
            "paths": cfg.assets.allow_paths,
            "uploads": cfg.assets.upload_max_bytes > 0,
            "upload_endpoint": "PUT /v1/assets/{sha256}",
            "upload_max_bytes": cfg.assets.upload_max_bytes,
            "upload_store_bytes": cfg.assets.upload_store_bytes,
            "upload_ttl_hours": cfg.assets.upload_ttl_hours,
        },
        "note": "2K output needs MiniMax's closed regeneration stage; local output is 768p.",
    }


@app.put("/v1/assets/{sha256}")
async def put_asset(sha256: str, request: Request) -> JSONResponse:
    """Take one attachment's bytes and keep them under the name their own content gives them.

    The way in for a caller that does not share a filesystem with this service. Named by the sha256
    of the file being sent, which makes the whole thing idempotent: sending the same clip twice
    stores it once, and a client that keeps a big reference in its graph pays the transfer on the
    first compile and never again.

    That name is a claim and it is treated as one. The bytes are hashed as they stream past and the
    file is placed under the digest computed here, so nothing a caller writes decides a path. A
    request whose bytes do not hash to their name is refused with the two things that cause it: a
    file that changed while it was being read, or a transfer that was cut short.

    Streamed to disk in chunks. The whole point is a video that would not be worth uploading if it
    had to fit in memory twice on the way.
    """
    # Every call into the store is inside the try, including the one that only looks. MEASURED: with
    # the lookup above it, a name that is not a digest -- `ref1.png`, 63 characters, anything with a
    # null byte in it -- raised out of the handler as an empty HTTP 500, which is the one shape of
    # failure a caller can do nothing with at all.
    declared = request.headers.get("content-length")
    try:
        held = uploads.stored(sha256)
        if held is not None:
            # Already here, so these bytes are not needed -- the stored file provably hashes to this
            # name, whatever this request is carrying. Drained rather than left unread, because a
            # server that answers without reading the body can have the rest of it land on a closed
            # socket, turning a success into a connection error at the other end.
            async for _ in request.stream():
                pass
            uploads.touch(held)
            return JSONResponse(status_code=200, content={
                "sha256": sha256, "stored": True, "bytes": held.stat().st_size,
                "note": "already held; the bytes sent were not needed"})
        with uploads.receiver(sha256, declared=int(declared) if declared else None) as rx:
            async for chunk in request.stream():
                rx.write(chunk)
            _, size = rx.finish()
    except uploads.NotADigest as e:
        await _drain(request)
        raise HTTPException(422, detail={"code": "asset-name-not-a-digest",
                                         "message": str(e)}) from e
    except uploads.TooLarge as e:
        await _drain(request)
        raise HTTPException(413, detail={"code": "asset-too-large", "message": str(e)}) from e
    except uploads.StoreFull as e:
        await _drain(request)
        raise HTTPException(507, detail={"code": "upload-store-full", "message": str(e)}) from e
    except uploads.DigestMismatch as e:
        raise HTTPException(422, detail={"code": "asset-digest-mismatch",
                                         "message": str(e)}) from e
    return JSONResponse(status_code=201, content={"sha256": sha256, "stored": True, "bytes": size})


async def _drain(request: Request, budget: int = 0) -> None:
    """Read off the rest of a body we are about to refuse, so the refusal can be read.

    MEASURED, and the reason this exists: answering while the client is still sending closes the
    connection under its own write, and the sentence naming the real problem is replaced at the far
    end by a broken pipe. Two uploads of identical size in one run got different treatment -- one the
    503-shaped message it deserved and one a socket error -- because whether it happens depends on
    what fits in a buffer, which is the worst kind of difference to debug.

    Bounded, because reading a body without limit is the exhaustion this endpoint's caps exist to
    prevent: a caller told the ceiling and sending fifty gigabytes anyway gets the connection closed
    under it, and that is the correct outcome for a caller that is not listening.
    """
    from starlette.requests import ClientDisconnect

    budget = budget or max(8 * 1024 * 1024, get_config().assets.upload_max_bytes + (1 << 20))
    seen = 0
    try:
        async for chunk in request.stream():
            seen += len(chunk)
            if seen > budget:
                return
    except ClientDisconnect:
        return


@app.post("/v1/briefs")
def create_brief(body: BriefIn) -> JSONResponse:
    brief = _to_brief(body)
    opts = ProfileOptions(name=get_config().profile)
    try:
        doc = compile_brief(brief, opts=opts, seed=body.seed,
                            llm=(body.compile_mode != "draft_only"),
                            thinking_prose=(body.effort == "max"),
                            transcripts=dict(body.transcripts))
    except BriefRefused as e:
        # Every refusal this layer makes about the request itself, with the code it carries. The
        # capacity one is design.md 12's, and 422 rather than a silently truncated manifest: which
        # reference matters is the caller's call, not ours.
        raise HTTPException(422, detail={"code": e.code, "message": str(e)}) from e
    except BackendUnavailable as e:
        raise HTTPException(503, detail={"code": "llm-unavailable", "message": str(e)}) from e
    except BackendError as e:
        raise HTTPException(502, detail={"code": "llm-error", "message": str(e)}) from e
    # The analyser writes its refusals for a person to read, and nothing caught them: a corrupt
    # clip, a still attached as `kind: video`, or a machine with no ffmpeg all reached the caller
    # as `Internal Server Error` with an empty body, and every one of those carefully worded
    # messages was discarded. Subclass first: a missing binary is this deployment's problem and
    # not something the caller can correct, so it keeps the 503 shape the LLM outage already uses.
    except ToolMissing as e:
        raise HTTPException(503, detail={"code": "analysis-tool-missing",
                                         "message": str(e)}) from e
    except AssetAnalysisError as e:
        raise HTTPException(422, detail={"code": "asset-unreadable", "message": str(e)}) from e

    brief_id = uuid.uuid4().hex[:16]
    _remember(brief_id, brief, doc)
    env = _envelope(brief_id, doc, brief)
    if not doc.ok:
        # Contradictions and internal invariants are separated: a caller can act on the first
        # and should never see the second, so an internal failure is reported as ours.
        caller_facing = [f for f in doc.errors if f.rule.split("-")[0] in
                         ("T6", "T7", "M5", "M6", "M7", "L3", "X10", "X11", "X12", "X13", "X15")]
        env["errors"] = [{"rule": f.rule, "message": f.msg} for f in doc.errors]
        env["status"] = "invalid"
        return JSONResponse(status_code=422 if caller_facing else 500, content=env)
    return JSONResponse(status_code=201, content=env)


@app.get("/v1/briefs/{brief_id}")
def get_brief(brief_id: str) -> dict[str, Any]:
    rec = _STORE.get(brief_id)
    if not rec:
        raise HTTPException(404, detail={"code": "unknown-brief", "message": brief_id})
    return _envelope(brief_id, rec["doc"], rec["brief"])


@app.patch("/v1/briefs/{brief_id}")
def refine_brief(brief_id: str, body: RefineIn) -> JSONResponse:
    rec = _STORE.get(brief_id)
    if not rec:
        raise HTTPException(404, detail={"code": "unknown-brief", "message": brief_id})
    if not body.change.strip():
        # An empty change used to return 200, report `changed: []`, bump the version and burn a full
        # recompile: a model call, a re-analysis and a new document, to apply nothing. Say so
        # instead, before spending any of it.
        raise HTTPException(422, detail={
            "code": "change-empty",
            "message": "`change` is empty, so there is nothing to apply. Say what to change in "
                       "plain language, for example 'make it 8 seconds' or 'lose the dialogue'."})
    try:
        doc, amended, changed = refine(rec["brief"], body.change,
                                       opts=ProfileOptions(name=get_config().profile))
    except BriefRefused as e:
        raise HTTPException(422, detail={"code": e.code, "message": str(e)}) from e
    except BackendUnavailable as e:
        raise HTTPException(503, detail={"code": "llm-unavailable", "message": str(e)}) from e
    except BackendError as e:
        raise HTTPException(502, detail={"code": "llm-error", "message": str(e)}) from e
    # Refining recompiles, so it reaches the analyser too -- with the same assets, which means an
    # asset that has become unreadable since the first compile lands here rather than there.
    except ToolMissing as e:
        raise HTTPException(503, detail={"code": "analysis-tool-missing",
                                         "message": str(e)}) from e
    except AssetAnalysisError as e:
        raise HTTPException(422, detail={"code": "asset-unreadable", "message": str(e)}) from e
    _remember(brief_id, amended, doc)
    env = _envelope(brief_id, doc, amended)
    env["changed"] = changed
    env["version"] = _STORE[brief_id]["versions"]
    return JSONResponse(status_code=200, content=env)


@app.get("/v1/briefs/{brief_id}/prompt")
def get_prompt(brief_id: str) -> dict[str, Any]:
    """The one field the renderer needs, for a caller that already has its own wiring."""
    rec = _STORE.get(brief_id)
    if not rec:
        raise HTTPException(404, detail={"code": "unknown-brief", "message": brief_id})
    doc: IRDocument = rec["doc"]
    # The retention marker the brief asserts about each attachment, keyed by the label it was
    # asserted about. A caller that wired the files cannot read this off the prose it did not write,
    # and it is the difference between "the file arrived" and "the file arrived as the thing you
    # meant": an attachment the brief marks weak_reference is not going to be reproduced.
    marker = {src: s.retention for s in doc.plan.subjects for src in s.sources}
    return {"prompt": doc.prompt, "mode": doc.mode.value,
            "wiring": [{"label": m.label, "wiring": m.wiring, "sha256": m.sha256,
                        "kind": m.kind.value, "sizing": m.sizing,
                        "retention": marker.get(m.label, "")}
                       for m in doc.plan.manifest],
            "frames": doc.plan.target.frames, "canvas": list(doc.plan.target.canvas),
            "render_hash": doc.render_hash()}
