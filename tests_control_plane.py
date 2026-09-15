# -*- coding: utf-8 -*-
"""regression_suite.py — 今日六 bug 永久回归测试（总监已批"全部变成 regression tests"）
纯 HTTP 打服务，零 GPU、零模型生成调用（卡片全缓存）。任一失败=退出码 1。
R1 同 spec 两次编译 byte-identical
R2 provenance 一致性（requested=draft_only ∧ prose_attempted=False ∧ source=draft）
R3 camera 纯度：三档各恰 1 条相机句；Static 无运动动词（硬闸曾放过 push——回归点）
R4 speaker_subject：双台词绑不同 Subject（render append 路径回归）
R5 anchor 帧级：spec_frame 精确 + 句在 prompt
R6 controlled 无 camera_phrase 必须拒（fail-loud，不许静默回落轮换）
R7 未知字段拒绝（extra=forbid 契约）
"""
import json, sys, urllib.request

API = "http://127.0.0.1:8420"
sys.path.insert(0, r"C:/Users/ghb/cpwm/work/_tmp/v1_trace/pipeline")
sys.path.insert(0, r"C:/Users/ghb/cpwm/projects/linwan/gate2")
from h3ir_http import shot_to_request
from spec_l0_jia import SPEC

fails = []
def check(name, ok, detail=""):
    print(("✓" if ok else "✗"), name, detail[:150])
    if not ok:
        fails.append(name)

def post(payload):
    req = urllib.request.Request(f"{API}/v1/briefs",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"content-type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read()), None
    except urllib.error.HTTPError as e:
        return None, e.code

import re
ANCH = [
    {"anchor_id": "A01", "subject": 1, "event": "faces the camera directly", "seconds": 0.5},
    {"anchor_id": "A02", "subject": 1, "event": "has turned 10-20 degrees left", "seconds": 1.5},
]

base_payload = shot_to_request(SPEC)
base_payload.update(seconds=124/24.0, compile_mode="draft_only", mode="ref2va", shots=1,
                    scene_text=SPEC["dd"], action_anchors=ANCH,
                    camera_phrase={"type": "Static Shot", "amplitude": None, "speed": None},
                    controlled=True)

# R1/R2/R3/R5
d1, _ = post(base_payload)
d2, _ = post(base_payload)
p1 = d1["ir"]["prompt"]; p2 = d2["ir"]["prompt"]
check("R1 双编译 byte-identical", p1 == p2)
pv = d1["ir"]["provenance"]
check("R2 provenance 一致", pv.get("requested_compile_mode") == "draft_only"
      and pv.get("prose_attempted") is False and d1.get("source") == "draft")
cams = re.findall(r"The camera [^.]*\.", p1)
check("R3a 相机句恰1条", len(cams) == 1, str(cams))
motion_words = ("push", "zoom", "pan", "truck", "tilt", "pedestal", "arc", "track", "shake", "roll")
check("R3b Static 硬闸（曾漏 push）", not any(w in cams[0].lower() for w in motion_words)
      and "static" in cams[0].lower(), cams[0] if cams else "")
tr = pv.get("action_anchor_trace") or []
frames = {x["anchor_id"]: x["spec_frame"] for x in tr}
check("R5 锚点帧级", frames.get("A01") == 12 and frames.get("A02") == 36
      and all(x["prompt_span"] in p1 for x in tr), str(frames))

# R4 双台词不同 Subject
# 双人板（two-subject sheet），否则 Subject2 越界回落——测例必须配双人素材
DUAL = {**SPEC,
        "subjects": [{"subject_id": "twoshot",
                      "raw": "two people standing in a plain indoor room matching the reference plate",
                      "assets": ["fnd004/plate_womanLeft.png"]}],
        "dd": ("Two people stand side by side matching the reference plate, facing camera, no contact. "
               "The woman on the left speaks first, then the man on the right answers. "
               "Locked camera, Static Shot, plain indoor background, for the entire clip."),
        "dialogue": [
            {"text": "第一句测试", "lang": "zh", "speaker_label": "the woman", "speaker_subject": 1},
            {"text": "第二句测试", "lang": "zh", "speaker_label": "the man", "speaker_subject": 2}],
        "audio_policy": {"expected_speech": True}}
sp = shot_to_request(DUAL)
sp.update(seconds=124/24.0, compile_mode="draft_only", mode="ref2va", shots=1,
          scene_text=DUAL["dd"],
          camera_phrase={"type": "Static Shot", "amplitude": None, "speed": None},
          controlled=True)
d4, _ = post(sp)
p4 = d4["ir"]["prompt"]
pairs = re.findall(r"<Subject (\d)> \(S(\d)\) says", p4)
check("R4 双台词绑不同 Subject", len({a for a, b in pairs}) == 2, str(pairs))

# R6 controlled 缺 camera_phrase → 必须非 201
bad = dict(base_payload); bad.pop("camera_phrase")
d6, code6 = post(bad)
check("R6 controlled 无相机参数被拒", d6 is None and code6 in (422, 500), f"HTTP {code6}")

# R7 未知字段
worse = dict(base_payload); worse["unknown_field_xyz"] = 1
d7, code7 = post(worse)
check("R7 未知字段拒绝", d7 is None and code7 == 422, f"HTTP {code7}")

print("\n回归套结果:", "全过 ✅" if not fails else f"失败: {fails}")
sys.exit(1 if fails else 0)
