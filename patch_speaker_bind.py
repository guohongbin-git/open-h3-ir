# -*- coding: utf-8 -*-
"""patch_speaker_bind.py — 编译层控制平面修复：dialogue 可按 Subject 序号指定说话人
总监 doctrine（camera_phrase 两次批准的同一原则）：控制平面归调用方。
根因：build_speakers 把 S1→people[0] 按位置绑定，单条台词永远移不到 Subject 2。
修复：DialogueIn/DialogueLine 加 speaker_subject（1-based Subject 序），
build_speakers 有值则绑定 people[idx-1]；无值保持原位置行为（兼容）。
"""
from pathlib import Path
import ast

F = Path("h3ir")

# 1) models.DialogueLine 加字段
m = F / "models.py"
mt = m.read_text(encoding="utf-8")
old = '''    text: str
    language: str = "English"
    speaker_hint: str | None = None
    voiceover: bool = False'''
new = '''    text: str
    language: str = "English"
    speaker_hint: str | None = None
    voiceover: bool = False
    speaker_subject: int | None = None   # 1-based <Subject N>; caller-owned binding'''
assert mt.count(old) == 1
m.write_text(mt.replace(old, new), encoding="utf-8")

# 2) plan.build_speakers 用 speaker_subject
p = F / "plan.py"
pt = p.read_text(encoding="utf-8")
old2 = '''        sid = f"(S{len(out) + 1})"
        subj = people[len(out)].label if len(out) < len(people) else (
            people[0].label if people else None)'''
new2 = '''        sid = f"(S{len(out) + 1})"
        # Caller-owned subject binding (director doctrine, cf. camera_phrase): an explicit
        # 1-based speaker_subject binds the line to that <Subject N>; absent it, keep the
        # positional behavior. Without this a single dialogue line can never move off
        # people[0], which made two-person speaker tests structurally contradictory.
        if line.speaker_subject and 1 <= line.speaker_subject <= len(people):
            subj = people[line.speaker_subject - 1].label
        else:
            subj = people[len(out)].label if len(out) < len(people) else (
                people[0].label if people else None)'''
assert pt.count(old2) == 1
p.write_text(pt.replace(old2, new2), encoding="utf-8")

# 3) service.DialogueIn + _to_brief 透传
s = F / "service.py"
st = s.read_text(encoding="utf-8")
old3 = '''class DialogueIn(BaseModel):'''
# 找其字段块（speaker: str | None = None）
old3b = '''    speaker: str | None = None'''
new3b = '''    speaker: str | None = None
    speaker_subject: int | None = Field(
        None, description="1-based <Subject N> that owns this line (caller-owned speaker "
                          "binding; required for multi-person speaker-selection exams).")'''
assert st.count(old3b) == 1
st = st.replace(old3b, new3b)
old4 = '''                                        speaker_hint=d.speaker, voiceover=d.voiceover)'''
new4 = '''                                        speaker_hint=d.speaker, voiceover=d.voiceover,
                                        speaker_subject=d.speaker_subject)'''
assert st.count(old4) == 1
st = st.replace(old4, new4)
s.write_text(st, encoding="utf-8")

# 4) contract DIALOGUE_FIELDS + 版本
ct = F / "contract.py"
c = ct.read_text(encoding="utf-8")
c = c.replace('DIALOGUE_FIELDS: tuple[str, ...] = ("text", "language", "speaker", "voiceover")',
              'DIALOGUE_FIELDS: tuple[str, ...] = ("text", "language", "speaker", "voiceover", "speaker_subject")')
c = c.replace("CONTRACT_VERSION = 4", "CONTRACT_VERSION = 5")
ct.write_text(c, encoding="utf-8")

for mod in ("models", "plan", "service", "contract"):
    ast.parse((F / f"{mod}.py").read_text(encoding="utf-8"))
print("speaker 绑定修复完成，四文件语法 OK，contract v5")
