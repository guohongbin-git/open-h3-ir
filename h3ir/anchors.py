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
