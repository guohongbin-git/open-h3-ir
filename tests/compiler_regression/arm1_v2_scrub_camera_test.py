# -*- coding: utf-8 -*-
"""RENDER-BRANCH A1 沉淀回归：v2 scrub（M3自生成锚句清除）+ 相机句恰1。"""
import sys, re
sys.path.insert(0, r"C:/Users/ghb/cpwm/projects/linwan/gate2")
from h3ir_http import arm1_render_prompt_v2
NL = chr(10)
def test_v2_scrub_camera():
    m3 = ("detailed_description:" + NL +
          "[Shot 1] At 00:02.100, <Subject 1> turns on her own somehow. She turns slowly." + NL + NL +
          "overall_soundscape:" + NL + "room tone.")
    anchors = [{"subject": 1, "event": "faces camera", "seconds": 0.5}]
    cam = "The camera holds a static shot."
    out = arm1_render_prompt_v2(m3, anchors, cam)
    assert "At 00:02.100" not in out, "M3 自生成锚句未被 scrub"
    assert out.count("At 00:00.500") == 1
    assert len(re.findall(re.escape(cam.rstrip(".")), out, re.I)) == 1, "相机句非恰1"
    assert arm1_render_prompt_v2(out, anchors, cam) == out, "幂等失败"
    print("OK arm1_v2 scrub+camera+幂等")

if __name__ == "__main__":
    test_v2_scrub_camera()
