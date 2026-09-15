# -*- coding: utf-8 -*-
"""patch_camera_control.py — fork 补丁：camera_phrase 一等参数 + 控制平面纯度门
总监裁决版（2026-09-14，CRITICAL for experimental validity）：
- production 模式：camera_phrase=None → 保持 DRAFT_CAMERA 轮换（兼容旧调用）
- controlled 模式：camera_phrase 必填，缺失 = 编译失败
- 三不变量：①最终 prompt 恰含 1 条相机指令 ②反解析 == camera_phrase ③无其它轮换派生句
- Static 硬闸：语义 Static 的文字里不得出现任何运动动词
"""
from pathlib import Path

F = Path("h3ir")

# ---------- anchors.py 增补：反解析 + 纯度门（自带 import re） ----------
ADDITION = '''

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
'''
a = F / "anchors.py"
at = a.read_text(encoding="utf-8")
if "camera_purity_check" not in at:
    a.write_text(at.rstrip() + ADDITION, encoding="utf-8")
    print("anchors.py +相机纯度门")

# ---------- draft.py：camera_phrase 替换轮换 ----------
d = F / "draft.py"
dt = d.read_text(encoding="utf-8")
old_sig = '''def deterministic_draft(brief: Brief, mode: Mode, cards: dict[str, AssetCard], *,
                        opts: ProfileOptions | None = None, loras=None, mode_decision=None,
                        licence=None):'''
new_sig = '''def deterministic_draft(brief: Brief, mode: Mode, cards: dict[str, AssetCard], *,
                        opts: ProfileOptions | None = None, loras=None, mode_decision=None,
                        licence=None, camera_phrase: dict | None = None):'''
assert dt.count(old_sig) == 1
dt = dt.replace(old_sig, new_sig)
old_rot = "    rotation = draft_camera(MAGNITUDE[parse(brief.creativity)])"
new_rot = '''    # Controlled experiment (director ruling 2026-09-14): the caller's camera_phrase
    # replaces the rotation wholesale; production (None) keeps the template behavior.
    if camera_phrase is not None:
        rotation = [dict(camera_phrase)]
    else:
        rotation = draft_camera(MAGNITUDE[parse(brief.creativity)])'''
assert dt.count(old_rot) == 1
dt = dt.replace(old_rot, new_rot)
d.write_text(dt, encoding="utf-8")
print("draft.py +camera_phrase 通道")

# ---------- compile.py：参数 + controlled 必填 + 纯度门 ----------
c = F / "compile.py"
ct = c.read_text(encoding="utf-8")
old = '''                  scene_text: str | None = None) -> IRDocument:'''
new = '''                  scene_text: str | None = None,
                  camera_phrase: dict[str, Any] | None = None,
                  controlled: bool = False) -> IRDocument:'''
assert ct.count(old) == 1
ct = ct.replace(old, new)

old2 = "        check_request(brief)\n        check_capacity(brief)"
new2 = '''        check_request(brief)
        check_capacity(brief)
        # Controlled-experiment gate: the camera control plane must be caller-owned and
        # never silently fall back to the draft rotation (CRITICAL for experimental validity).
        if controlled and camera_phrase is None:
            raise BriefRefused("camera-phrase-required",
                               "controlled 模式必须提供 camera_phrase（type/amplitude/speed），"
                               "否则运镜会悄悄落回 DRAFT_CAMERA 轮换，污染单变量条件。")'''
assert ct.count(old2) == 1
ct = ct.replace(old2, new2)

old3 = '''        draft_plan = deterministic_draft(brief, mode, cards, opts=opts, loras=loras,
                                         mode_decision=mode_decision)'''
new3 = '''        draft_plan = deterministic_draft(brief, mode, cards, opts=opts, loras=loras,
                                         mode_decision=mode_decision,
                                         camera_phrase=camera_phrase)'''
assert ct.count(old3) == 1
ct = ct.replace(old3, new3)

old4 = '''        if not llm:
            # Arm 0: the deterministic draft IS the deliverable.'''
new4 = '''        if camera_phrase is not None:
            from .anchors import camera_purity_check
            bad = camera_purity_check(draft_result.prompt, camera_phrase)
            if bad:
                raise CompilerInvariantError("相机控制平面纯度门失败: " + "; ".join(bad))
        if not llm:
            # Arm 0: the deterministic draft IS the deliverable.'''
assert ct.count(old4) == 1
ct = ct.replace(old4, new4)
c.write_text(ct, encoding="utf-8")
print("compile.py +controlled 必填 + 纯度门")

# ---------- service.py：BriefIn 字段 + 透传 ----------
s = F / "service.py"
st = s.read_text(encoding="utf-8")
old5 = "    transcripts: dict[str, str] = Field("
new5 = '''    camera_phrase: dict[str, Any] | None = Field(
        None,
        description="Caller-owned camera directive {type, amplitude(small|large|null), "
                    "speed(slow|fast|null)} from the official closed motion vocabulary. In "
                    "controlled experiments it REPLACES the draft camera rotation; the "
                    "purity gate proves the final prompt carries exactly this one directive.")
    controlled: bool = Field(
        False,
        description="Controlled-experiment mode: camera_phrase becomes mandatory and the "
                    "purity gate is a hard compile failure. Production (False) keeps "
                    "template rotation when camera_phrase is absent.")
    transcripts: dict[str, str] = Field('''
assert st.count(old5) == 1
st = st.replace(old5, new5, 1)
old6 = '''                            action_anchors=body.action_anchors,
                            scene_text=body.scene_text)'''
new6 = '''                            action_anchors=body.action_anchors,
                            scene_text=body.scene_text,
                            camera_phrase=body.camera_phrase,
                            controlled=body.controlled)'''
assert st.count(old6) == 1
st = st.replace(old6, new6)
s.write_text(st, encoding="utf-8")
print("service.py +camera_phrase/controlled 字段与透传")

# ---------- contract ----------
p = F / "contract.py"
pt = p.read_text(encoding="utf-8")
pt = pt.replace('"action_anchors", "scene_text",',
                '"action_anchors", "scene_text", "camera_phrase", "controlled",')
pt = pt.replace("CONTRACT_VERSION = 3", "CONTRACT_VERSION = 4")
p.write_text(pt, encoding="utf-8")
print("contract.py v4 字段发布")

import ast
for m in ("anchors", "draft", "compile", "service", "contract"):
    ast.parse((F / f"{m}.py").read_text(encoding="utf-8"))
print("五文件语法全 OK")
