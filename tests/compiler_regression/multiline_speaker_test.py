# -*- coding: utf-8 -*-
"""总监回归缺口②：多台词跨 Subject 绑定——经服务真实编译验证（原 append 路径 bug 案）。"""
import json, re, sys, urllib.request
sys.path.insert(0, r"C:/Users/ghb/cpwm/work/_tmp/v1_trace/pipeline")
sys.path.insert(0, r"C:/Users/ghb/cpwm/projects/linwan/gate2")
from h3ir_http import shot_to_request

def test_multiline_speaker():
    spec = {"id": "REG-multiline", "aspect": "21:9", "frames": 124, "shots": 1,
            "subjects": [{"subject_id": "twoshot",
                          "raw": "two people standing in a plain indoor room matching the reference plate",
                          "assets": ["fnd004/plate_womanLeft.png"]}],
            "dd": ("Two people stand side by side matching the reference plate, facing camera, no contact. "
                   "The woman on the left speaks first, then the man on the right answers. "
                   "Locked camera, Static Shot, plain indoor background, for the entire clip."),
            "dialogue": [
                {"text": "甲句测试", "lang": "zh", "speaker_label": "the woman", "speaker_subject": 1},
                {"text": "乙句测试", "lang": "zh", "speaker_label": "the man", "speaker_subject": 2}],
            "audio_policy": {"expected_speech": True},
            "sound_ambience": "quiet room tone", "music": "None."}
    payload = shot_to_request(spec)
    payload.update(seconds=124/24.0, compile_mode="draft_only", mode="ref2va", shots=1,
                   scene_text=spec["dd"],
                   camera_phrase={"type": "Static Shot", "amplitude": None, "speed": None},
                   controlled=True)
    req = urllib.request.Request("http://127.0.0.1:8420/v1/briefs",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"content-type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    pairs = re.findall(r"<Subject (\d)> \((S\d)\) says", d["ir"]["prompt"])
    assert len(pairs) == 2, f"台词渲染数 {len(pairs)}（应 2）"
    subjects = {a for a, b in pairs}
    assert len(subjects) == 2, f"双台词绑同一 Subject {pairs}（append 路径 bug 复发）"
    print("✓ multiline_speaker_test（双台词分绑两主体）")

if __name__ == "__main__":
    test_multiline_speaker()
