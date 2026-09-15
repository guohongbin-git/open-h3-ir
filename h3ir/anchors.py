# -*- coding: utf-8 -*-
"""anchors.py — 镜内动作锚点的确定性注入（Arm 0 计划 rev2 任务 5）

设计（GPT 审查定稿）：
- 锚点是调用方的一等结构字段，不是 prose：Python 机械生成句子，模型永不触碰。
- 时间一律以帧为准（spec_frame），canonical_time = frame/24 只是显示值。
- 注入后 shot.body = 锚点句序列（完全确定），draft 的 beat 文本被替换。
- 第一镜头部垫无时间引导句（validator T2：[Shot 1] 头不许带时间戳）。
- sidecar（anchor trace）记录 spec_frame / compiled_span / preservation，供门验证。
"""
from __future__ import annotations

import math
from typing import Any

FPS = 24.0


def _ts(frame: int) -> str:
    """帧 → 官方 At MM:SS.mmm 时间戳（帧是唯一真值，时间是显示）。"""
    s = frame / FPS
    mm, ss = divmod(s, 60)
    return f"{int(mm):02d}:{ss:06.3f}"


def normalise(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验+量化调用方锚点：seconds→frame（最近帧），补 canonical_time/anchor_id。"""
    out = []
    for i, a in enumerate(raw, start=1):
        if "frame" in a:
            f = int(a["frame"])
        else:
            f = int(math.floor(float(a["seconds"]) * FPS + 0.5))
        out.append({
            "anchor_id": a.get("anchor_id") or f"A{i:02d}",
            "subject": int(a.get("subject", 1)),
            "event": str(a["event"]).strip().rstrip("."),
            "spec_frame": f,
            "canonical_time": round(f / FPS, 6),
            "preservation": "exact",
        })
    out.sort(key=lambda x: (x["spec_frame"], x["anchor_id"]))
    return out


def render_sentences(anchors: list[dict[str, Any]]) -> str:
    """锚点 → 官方 At 时间戳句子序列。逐字确定，同一输入永远同一输出。"""
    parts = []
    for a in anchors:
        parts.append(f"At {_ts(a['spec_frame'])}, <Subject {a['subject']}> {a['event']}.")
    return " ".join(parts)


def inject(plan, anchors_raw: list[dict[str, Any]],
         scene_text: str | None = None) -> list[dict[str, Any]]:
    """把锚点注进 draft plan：单镜 = 全部注入 shot 1；多镜按帧窗落入对应 shot。
    返回 anchor trace（sidecar 数据）。"""
    anchors = normalise(anchors_raw) if anchors_raw else []
    if not plan.shots:
        return []
    if not anchors and scene_text and plan.shots:
        plan.shots[0].body = ("The clip is a single continuous take of the scene described "
                              "below. " + scene_text.strip())
        return [{"anchor_id": "SCENE", "spec_frame": 0, "compiled_frame": 0,
                 "shot": 1, "prompt_span": scene_text.strip()[:80],
                 "preservation": "SEMANTIC"}]
    spans: list[dict[str, Any]] = []
    n = len(plan.shots)
    total = max(a["spec_frame"] for a in anchors) + 1
    bounds = [(i * total // n, (i + 1) * total // n) for i in range(n)]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for a in anchors:
        idx = n - 1 if a["spec_frame"] >= bounds[-1][1] else next(
            i for i, (s, e) in enumerate(bounds) if s <= a["spec_frame"] < e)
        grouped.setdefault(idx, []).append(a)
    for idx, shot in enumerate(plan.shots):
        group = grouped.get(idx)
        if not group:
            continue
        # scene_text (dd prose) rides before the anchor sentences: the caller's staging is
        # authoritative, so the body carries it verbatim instead of "the scene described by
        # the request" — the draft template drops dd semantics otherwise.
        parts = []
        if idx == 0 and scene_text:
            parts.append(scene_text.strip())
        parts.append(render_sentences(group))
        shot.body = " ".join(parts)
        for a in group:
            spans.append({"anchor_id": a["anchor_id"], "spec_frame": a["spec_frame"],
                          "compiled_frame": a["spec_frame"],
                          "shot": idx + 1,
                          "prompt_span": f"At {_ts(a['spec_frame'])}, "
                                         f"<Subject {a['subject']}> {a['event']}.",
                          "preservation": "exact"})
    return spans

# ===== 相机控制平面（总监裁决 2026-09-14：模板越权注入修复） =====
import re

CAMERA_CANONICAL = [
    "zooms in", "zooms out", "pushes in", "pulls out", "pans left", "pans right",
    "trucks left", "trucks right", "tilts up", "tilts down",
    "rises on a pedestal move", "lowers on a pedestal move",
    "arcs around the subject", "tracks with the subject", "holds a static shot",
    "shakes slightly", "shakes strongly", "takes the subject's point of view",
    "rolls clockwise", "rolls counterclockwise",
]
_CAM_ALT = "|".join(re.escape(v) for v in CAMERA_CANONICAL)
CAM_RE = re.compile(rf"The camera (?P<verb>{_CAM_ALT})(?: with (?P<amp>small|large) amplitude)?"
                    rf"(?: at (?P<spd>slow|fast) speed)?")
_VERB2TYPE = {
    "zooms in": "Zoom In", "zooms out": "Zoom Out", "pushes in": "Push In", "pulls out": "Pull Out",
    "pans left": "Pan Left", "pans right": "Pan Right", "trucks left": "Truck Left",
    "trucks right": "Truck Right", "tilts up": "Tilt Up", "tilts down": "Tilt Down",
    "rises on a pedestal move": "Pedestal Up", "lowers on a pedestal move": "Pedestal Down",
    "arcs around the subject": "Arc Shot", "tracks with the subject": "Tracking Shot",
    "holds a static shot": "Static Shot", "shakes slightly": "Shake Slightly",
    "shakes strongly": "Shake Strongly", "takes the subject's point of view": "POV",
    "rolls clockwise": "Roll Clockwise", "rolls counterclockwise": "Roll Counterclockwise",
}
MOTION_VERBS = [v for v in CAMERA_CANONICAL if v != "holds a static shot"]


def camera_purity_check(prompt: str, camera_phrase: dict) -> list[str]:
    """三不变量 + Static 硬闸。返回违规列表，空=干净。"""
    found = [{"verb": m.group("verb"), "amp": m.group("amp"), "spd": m.group("spd")}
             for m in CAM_RE.finditer(prompt)]
    v = []
    if len(found) != 1:
        v.append(f"相机指令数={len(found)}，控制平面要求恰 1 条（双指令=污染）")
        return v
    got = found[0]
    want_type = camera_phrase["type"]
    if _VERB2TYPE.get(got["verb"]) != want_type:
        v.append(f"相机指令解析为 {_VERB2TYPE.get(got['verb'])!r}，要求 {want_type!r}")
    if (got["amp"] or None) != (camera_phrase.get("amplitude") or None):
        v.append(f"amplitude 解析 {got['amp']!r}，要求 {camera_phrase.get('amplitude')!r}")
    if (got["spd"] or None) != (camera_phrase.get("speed") or None):
        v.append(f"speed 解析 {got['spd']!r}，要求 {camera_phrase.get('speed')!r}")
    if want_type == "Static Shot" and any(mv in prompt for mv in MOTION_VERBS):
        v.append("Static 语义但文字夹运动动词（语义-文字分裂硬闸）")
    return v
