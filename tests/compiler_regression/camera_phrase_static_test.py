# -*- coding: utf-8 -*-
"""compiler_regression / camera_phrase_static_test — 门0 命名回归
断言：controlled+Static Shot 的编译产物①恰 1 条相机指令②反解析=Static Shot③全文无任何运动动词。
回归原点：draft 模板曾把 'pushes in' 静默追加进"静态"题正文（甲题污染案）。
"""
import json, re, sys, urllib.request

API = "http://127.0.0.1:8420"
MOTION = ("zoom in", "zoom out", "push", "pull", "pan", "truck", "tilt",
          "pedestal", "arc", "track", "shake", "roll", "pov")

def payload_for(scene_text, cp):
    return {"intent": scene_text, "assets": [], "seconds": 124 / 24.0, "aspect": "21:9",
            "compile_mode": "draft_only", "mode": "t2va", "shots": 1,
            "camera_phrase": cp, "controlled": True}

def test_camera_phrase_static():
    scene = ("A young woman stands still facing the camera in a plain room. "
             "Locked camera, Static Shot, for the entire clip.")
    req = urllib.request.Request(f"{API}/v1/briefs",
        data=json.dumps(payload_for(scene, {"type": "Static Shot", "amplitude": None, "speed": None})).encode(),
        headers={"content-type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    p = d["ir"]["prompt"]
    cams = re.findall(r"The camera [^.]*\.", p)
    assert len(cams) == 1, f"相机指令 {len(cams)} 条（应恰 1）：{cams}"
    assert "static" in cams[0].lower(), f"相机句非 Static：{cams[0]}"
    hits = [m for m in MOTION if m in p.lower()]
    assert not hits, f"Static 语义夹运动动词：{hits}"
    print("✓ camera_phrase_static_test")

if __name__ == "__main__":
    test_camera_phrase_static()
