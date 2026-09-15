# -*- coding: utf-8 -*-
"""总监回归缺口①：锚点句保留性——机械回填后锚点句数==N 且无重复（桥级测试）。"""
import sys, math
sys.path.insert(0, r"C:/Users/ghb/cpwm/projects/linwan/gate2")
from h3ir_http import arm1_render_prompt, _anchor_sentence

def test_anchor_preservation():
    anchors = [{"subject": 1, "event": f"pose {i}", "seconds": i * 0.5} for i in range(1, 7)]
    m3 = ("detailed_description:\n[Shot 1] She moves through a fluid sequence while the camera "
          "stays still and the light remains soft.\n\noverall_soundscape:\nroom tone.")
    out = arm1_render_prompt(m3, anchors)
    # 锚点数 == N
    for a in anchors:
        assert _anchor_sentence(a) in out, f"锚点丢失: {a['seconds']}"
    # 不重复：每个时间戳只出现一次
    stamps = [f"At {int((math.floor(a['seconds']*24+0.5)/24)//60):02d}:{(math.floor(a['seconds']*24+0.5)/24)%60:06.3f}" for a in anchors]
    for s in stamps:
        assert out.count(s) == 1, f"锚点重复: {s}"
    # M3 原句保留（回填=补充规格非替换）
    assert "fluid sequence" in out
    # 幂等
    assert arm1_render_prompt(out, anchors) == out
    print("✓ anchor_preservation_test（N=6 保留/唯一/原句在/幂等）")

if __name__ == "__main__":
    test_anchor_preservation()
