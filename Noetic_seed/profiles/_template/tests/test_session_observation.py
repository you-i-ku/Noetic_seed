"""Session.push_observation + 4 種 label format テスト。

INTEGRATION_POINTS.md §2.4 の仕様を網羅:
  - push_observation: metadata 保持 + messages 追記
  - _render_observation_label: 4 format (structured_compact/full/natural_ja/compact)
  - observation_time 自動補完 (None で現在時刻)
  - 空 content / actor 欠落 / 未知 format fallback
  - Session 初期化時の format 指定 + 動的切替

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_session_observation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.session import Session


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _obs_text(session: Session, idx: int = -1) -> str:
    """session.messages[idx] の content[0].text を取り出す helper。"""
    return session.messages[idx]["content"][0]["text"]


# ============================================================
# push_observation: 基本動作
# ============================================================

def test_push_observation_basic():
    print("== push_observation: messages と observations に積まれる ==")
    s = Session()
    s.push_observation(
        observed_channel="device", content="おはよー",
        source_action_hint="living_presence", observation_time="09:30",
    )
    return all([
        _assert(len(s.messages) == 1, "messages 1 件"),
        _assert(s.messages[0]["role"] == "user", "role=user"),
        _assert("おはよー" in _obs_text(s), "content 含む"),
        _assert(len(s.observations) == 1, "observations メタ 1 件"),
        _assert(s.observations[0]["observed_channel"] == "device",
                "metadata channel"),
    ])


def test_push_observation_empty_content_skipped():
    print("== push_observation: 空 content は skip ==")
    s = Session()
    s.push_observation(observed_channel="device", content="")
    s.push_observation(observed_channel="device", content="x")
    return all([
        _assert(len(s.messages) == 1, "1 件のみ (空はスキップ)"),
        _assert(len(s.observations) == 1, "metadata も 1 件"),
    ])


def test_push_observation_auto_time():
    print("== push_observation: observation_time=None で現在時刻補完 ==")
    s = Session()
    s.push_observation(observed_channel="device", content="hi")
    meta = s.observations[0]
    # HH:MM 形式 (5 文字、区切り ':')
    return all([
        _assert(isinstance(meta["observation_time"], str), "str"),
        _assert(len(meta["observation_time"]) == 5, "HH:MM 5 字"),
        _assert(":" in meta["observation_time"], "':' 含む"),
    ])


def test_push_user_text_vs_observation():
    print("== push_user_text は label 無し、push_observation は label 付き ==")
    s = Session()
    s.push_user_text("plain text")
    s.push_observation(observed_channel="device", content="obs text",
                       observation_time="12:00")
    plain = _obs_text(s, 0)
    obs = _obs_text(s, 1)
    return all([
        _assert(not plain.startswith("["), "plain に label なし"),
        _assert(obs.startswith("[obs"), "observation に label あり"),
        _assert(len(s.observations) == 1, "observations は obs 分のみ"),
    ])


# ============================================================
# 4 format label rendering
# ============================================================

def test_format_structured_compact():
    print("== format=structured_compact: [obs channel=X action=Y time=Z] ==")
    s = Session(observation_label_format="structured_compact")
    s.push_observation(
        observed_channel="device", content="hi",
        source_action_hint="output_display", observation_time="15:30",
    )
    text = _obs_text(s)
    return all([
        _assert(text.startswith("[obs "), "[obs で始まる"),
        _assert("channel=device" in text, "channel=device"),
        _assert("action=output_display" in text, "action=output_display"),
        _assert("time=15:30" in text, "time=15:30"),
        _assert("hi" in text, "本文含む"),
    ])


def test_format_structured_full():
    print("== format=structured_full: source_action まで含む ==")
    s = Session(observation_label_format="structured_full")
    s.push_observation(
        observed_channel="device", content="hi",
        source_action_hint="living_presence", observation_time="09:00",
    )
    text = _obs_text(s)
    return all([
        _assert(text.startswith("[observation "), "[observation で始まる"),
        _assert("channel=device" in text, "channel"),
        _assert("action=living_presence" in text, "action"),
        _assert("source_action=living_presence" in text,
                "source_action 明示"),
        _assert("time=09:00" in text, "time"),
    ])


def test_format_natural_ja():
    print("== format=natural_ja: [Xからの声 Z] ==")
    s = Session(observation_label_format="natural_ja")
    s.push_observation(
        observed_channel="device", content="おやすみ",
        actor="ent_yuu", observation_time="23:45",
    )
    text = _obs_text(s)
    return all([
        _assert(text.startswith("[ent_yuu"), "actor が speaker 優先"),
        _assert("からの声" in text, "'からの声' 含む"),
        _assert("23:45" in text, "time 含む"),
        _assert("おやすみ" in text, "本文含む"),
    ])


def test_format_natural_ja_no_actor():
    print("== format=natural_ja: actor 無しなら channel を speaker に ==")
    s = Session(observation_label_format="natural_ja")
    s.push_observation(
        observed_channel="elyth", content="通知",
        observation_time="10:00",  # actor なし
    )
    text = _obs_text(s)
    return all([
        _assert(text.startswith("[elyth"), "channel を speaker に"),
        _assert("からの声" in text, "'からの声' 含む"),
    ])


def test_format_compact():
    print("== format=compact: [obs X Z] の極短 ==")
    s = Session(observation_label_format="compact")
    s.push_observation(
        observed_channel="x", content="tweet",
        source_action_hint="x_post", observation_time="20:00",
    )
    text = _obs_text(s)
    return all([
        _assert(text.startswith("[obs x"), "channel のみの短縮"),
        _assert("20:00" in text, "time 含む"),
        _assert("action=" not in text, "action 非表示 (compact)"),
        _assert("tweet" in text, "本文"),
    ])


# ============================================================
# Format 切替 / unknown fallback
# ============================================================

def test_unknown_format_init_rejected():
    print("== 未知 format を init で渡すと ValueError ==")
    try:
        Session(observation_label_format="xxx")
        return _assert(False, "ValueError 期待")
    except ValueError as e:
        return all([
            _assert(True, "ValueError 発生"),
            _assert("xxx" in str(e), "メッセージに format 名含む"),
        ])


def test_unknown_format_dynamic_fallback():
    print("== 動的に未知 format に書き換え → structured_compact に fallback ==")
    s = Session()
    s.observation_label_format = "unknown_fmt"  # 属性直接書換
    s.push_observation(
        observed_channel="device", content="x",
        source_action_hint="living_presence", observation_time="00:00",
    )
    text = _obs_text(s)
    # fallback format は structured_compact
    return all([
        _assert(text.startswith("[obs channel="), "fallback → compact 形式"),
        _assert("action=living_presence" in text, "metadata 保持"),
    ])


def test_format_switch_between_pushes():
    print("== push 途中で format 切替: 切替前後で label 変わる ==")
    s = Session(observation_label_format="structured_compact")
    s.push_observation(
        observed_channel="device", content="first",
        observation_time="10:00",
    )
    s.observation_label_format = "natural_ja"
    s.push_observation(
        observed_channel="device", content="second",
        observation_time="10:05",
    )
    first = _obs_text(s, 0)
    second = _obs_text(s, 1)
    return all([
        _assert(first.startswith("[obs channel="), "前は structured_compact"),
        _assert(second.startswith("[device"), "後は natural_ja"),
        _assert(len(s.observations) == 2, "metadata 2 件保持"),
    ])


# ============================================================
# clear
# ============================================================

def test_clear_resets_both():
    print("== clear: messages と observations 両方リセット ==")
    s = Session()
    s.push_observation(observed_channel="device", content="x")
    s.push_user_text("y")
    s.clear()
    return all([
        _assert(s.messages == [], "messages 空"),
        _assert(s.observations == [], "observations 空"),
    ])


# ============================================================
# serialize 互換性: push_observation 後でも既存 API 動作
# ============================================================

def test_serialize_after_observation():
    print("== push_observation 後も serialize_for_* が動く (claw-code 互換) ==")
    s = Session()
    s.push_observation(
        observed_channel="device", content="hello",
        source_action_hint="living_presence", observation_time="08:00",
    )
    s.push_user_text("follow-up question")
    msgs_ant = s.serialize_for_anthropic()
    msgs_oa = s.serialize_for_openai()
    return all([
        _assert(len(msgs_ant) == 2, "anthropic 2 messages"),
        _assert(all(m["role"] == "user" for m in msgs_ant),
                "全 role=user"),
        _assert(len(msgs_oa) == 2, "openai 2 messages"),
        _assert("hello" in msgs_oa[0]["content"], "observation が OpenAI にも"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("基本: messages + observations", test_push_observation_basic),
        ("空 content は skip", test_push_observation_empty_content_skipped),
        ("time 自動補完", test_push_observation_auto_time),
        ("push_user_text vs observation", test_push_user_text_vs_observation),
        ("format: structured_compact", test_format_structured_compact),
        ("format: structured_full", test_format_structured_full),
        ("format: natural_ja (actor あり)", test_format_natural_ja),
        ("format: natural_ja (actor なし)", test_format_natural_ja_no_actor),
        ("format: compact", test_format_compact),
        ("init で未知 format → ValueError", test_unknown_format_init_rejected),
        ("動的に未知 format → fallback", test_unknown_format_dynamic_fallback),
        ("format 動的切替", test_format_switch_between_pushes),
        ("clear 両方リセット", test_clear_resets_both),
        ("serialize 互換性", test_serialize_after_observation),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
