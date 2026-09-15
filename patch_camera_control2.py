# -*- coding: utf-8 -*-
"""patch_camera_control2.py — 补齐 compile/service/contract（anchors/draft 已打）"""
from pathlib import Path
import ast

F = Path("h3ir")

c = F / "compile.py"
ct = c.read_text(encoding="utf-8")
assert "camera_phrase" not in ct, "compile.py 已打过?"

old = "                  scene_text: str | None = None) -> IRDocument:"
new = """                  scene_text: str | None = None,
                  camera_phrase: dict[str, Any] | None = None,
                  controlled: bool = False) -> IRDocument:"""
assert ct.count(old) == 1; ct = ct.replace(old, new)

old2 = "        check_request(brief)\n        check_capacity(brief)"
new2 = '''        check_request(brief)
        check_capacity(brief)
        # Controlled-experiment gate: the camera control plane must be caller-owned and
        # never silently fall back to the draft rotation (CRITICAL for experimental validity).
        if controlled and camera_phrase is None:
            raise BriefRefused("camera-phrase-required",
                               "controlled 模式必须提供 camera_phrase（type/amplitude/speed），"
                               "否则运镜会悄悄落回 DRAFT_CAMERA 轮换，污染单变量条件。")'''
assert ct.count(old2) == 1; ct = ct.replace(old2, new2)

old3 = """        draft_plan = deterministic_draft(brief, mode, cards, opts=opts, loras=loras,
                                         mode_decision=decision)"""
new3 = """        draft_plan = deterministic_draft(brief, mode, cards, opts=opts, loras=loras,
                                         mode_decision=decision,
                                         camera_phrase=camera_phrase)"""
assert ct.count(old3) == 1; ct = ct.replace(old3, new3)

old4 = """        if not llm:
            # Arm 0: the deterministic draft IS the deliverable."""
new4 = """        if camera_phrase is not None:
            from .anchors import camera_purity_check
            bad = camera_purity_check(draft_result.prompt, camera_phrase)
            if bad:
                raise CompilerInvariantError("相机控制平面纯度门失败: " + "; ".join(bad))
        if not llm:
            # Arm 0: the deterministic draft IS the deliverable."""
assert ct.count(old4) == 1; ct = ct.replace(old4, new4)
c.write_text(ct, encoding="utf-8")
print("compile.py +controlled 必填 + 纯度门")

s = F / "service.py"
st = s.read_text(encoding="utf-8")
assert "camera_phrase" not in st
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
assert st.count(old5) == 1; st = st.replace(old5, new5, 1)
old6 = """                            action_anchors=body.action_anchors,
                            scene_text=body.scene_text)"""
new6 = """                            action_anchors=body.action_anchors,
                            scene_text=body.scene_text,
                            camera_phrase=body.camera_phrase,
                            controlled=body.controlled)"""
assert st.count(old6) == 1; st = st.replace(old6, new6)
s.write_text(st, encoding="utf-8")
print("service.py +camera_phrase/controlled")

p = F / "contract.py"
pt = p.read_text(encoding="utf-8")
pt = pt.replace('"action_anchors", "scene_text",',
                '"action_anchors", "scene_text", "camera_phrase", "controlled",')
pt = pt.replace("CONTRACT_VERSION = 3", "CONTRACT_VERSION = 4")
p.write_text(pt, encoding="utf-8")
print("contract.py v4")

for m in ("anchors", "draft", "compile", "service", "contract"):
    ast.parse((F / f"{m}.py").read_text(encoding="utf-8"))
print("五文件语法 OK")
