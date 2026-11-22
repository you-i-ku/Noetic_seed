"""段階13 Phase 0.2: event_emitter (observer pattern hub)。

PLAN §18-B.2 literal: tool 実行直後 同期 emit + 並列 observer pattern。
controller / main.py 等の発火側は `fire_event(state, entry)` のみ呼ぶ、
受信側 (`_record_entry` で jsonl + state view sync、Phase 0.4 で
`check_raw_subjective_gap` 等) は `subscribe(observer)` で登録。

哲学整合: tool 実行 = 行動、entry = 観測。fire_event は両者を 1 つの「event」
として中央ハブから多方向に配信、subscribers は副作用責任を受け持つ。Active
Inference 整合 (action × observation 統一視点)、`feedback_action_observation_unified`
に対応。

非同期化はしない (Phase 0.2 lean): subscribers は同期発火、保存・rebuild の
ordering を保証。Phase 4 で background graph maintenance を入れる時に必要なら
非同期 observer を検討。

例外ポリシー (Codex review 0.2 WARN 反映): 1 observer の例外は他 observer の
発火を止めない (isolation) が、silent fail を避けるため traceback を必ず出力
する。`_record_entry` (永続化) のような critical observer も同 isolation 経路
で扱われるため、失敗が観測可能な形で console に残る。critical 失敗で fire_event
全体を止めたい場合は Phase 0.4 で `subscribe_critical(observer)` 等の拡張で
対応 (現状 lean、判断は将来観測材料に依存)。
"""
import traceback
from typing import Callable, List

# observer = (state, entry) を受ける callable。戻り値は無視。
Observer = Callable[[dict, dict], None]

_subscribers: List[Observer] = []


def subscribe(observer: Observer) -> None:
    """observer を登録。fire_event で同期発火される。重複登録は無視。"""
    if observer not in _subscribers:
        _subscribers.append(observer)


def unsubscribe(observer: Observer) -> None:
    """observer を解除。未登録なら何もしない (defensive)。"""
    if observer in _subscribers:
        _subscribers.remove(observer)


def fire_event(state: dict, entry: dict) -> None:
    """tool 実行直後の中央ハブ。登録 subscribers を順次同期発火。

    1 つの observer の例外は他 observer の発火を止めない (isolation)。
    例外は print のみで握りつぶす (Phase 0.4 で hooks.py 既存パターンに揃える
    可能性、現状 lean)。
    """
    # iteration 中の subscribe/unsubscribe で list 変化しないよう snapshot
    for observer in list(_subscribers):
        try:
            observer(state, entry)
        except Exception as exc:
            # Codex review 0.2 WARN 反映: silent fail 防止のため traceback も出力
            print(f"  [event_emitter] observer 例外 (skip): {exc}")
            traceback.print_exc()


def reset_subscribers() -> None:
    """test 用 cleanup。production では呼ばない。"""
    _subscribers.clear()


def list_subscribers() -> List[Observer]:
    """test / debug 用に現在の subscribers のコピーを返す。"""
    return list(_subscribers)
