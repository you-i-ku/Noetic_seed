"""段階13 Phase 0.2 — event_emitter (observer pattern hub) テスト。

PLAN §18-B.2: tool 実行直後 同期 emit + 並列 observer pattern。
- subscribe / fire_event / unsubscribe / reset_subscribers / list_subscribers
- 重複登録無視、例外 isolation、iteration 中の subscribe 安全性

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_event_emitter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import event_emitter


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup():
    """各 test 前に subscribers をリセット。"""
    event_emitter.reset_subscribers()


def test_subscribe_and_fire():
    print("== subscribe + fire_event: observer が呼ばれる ==")
    _setup()
    received = []

    def obs(state, entry):
        received.append((state, entry))

    event_emitter.subscribe(obs)
    state = {"x": 1}
    entry = {"id": "e1"}
    event_emitter.fire_event(state, entry)

    return all([
        _assert(len(received) == 1, "1 回発火"),
        _assert(received[0][0] is state, "state がそのまま渡る"),
        _assert(received[0][1] is entry, "entry がそのまま渡る"),
    ])


def test_multiple_observers_in_order():
    print("== 複数 observer: subscribe 順に発火 ==")
    _setup()
    order = []

    def obs1(state, entry):
        order.append(1)

    def obs2(state, entry):
        order.append(2)

    def obs3(state, entry):
        order.append(3)

    event_emitter.subscribe(obs1)
    event_emitter.subscribe(obs2)
    event_emitter.subscribe(obs3)
    event_emitter.fire_event({}, {"id": "e1"})

    return _assert(order == [1, 2, 3], f"順次発火: {order}")


def test_duplicate_subscribe_ignored():
    print("== 重複 subscribe: 同じ observer は 1 回のみ登録 ==")
    _setup()
    count = [0]

    def obs(state, entry):
        count[0] += 1

    event_emitter.subscribe(obs)
    event_emitter.subscribe(obs)
    event_emitter.subscribe(obs)
    event_emitter.fire_event({}, {})

    return all([
        _assert(len(event_emitter.list_subscribers()) == 1, "subscribers 1 個"),
        _assert(count[0] == 1, f"発火 1 回 (got {count[0]})"),
    ])


def test_exception_isolation():
    print("== 例外 isolation: 1 observer 失敗で他 observer 継続 ==")
    _setup()
    after = []

    def obs_fail(state, entry):
        raise RuntimeError("intentional failure")

    def obs_after(state, entry):
        after.append(1)

    event_emitter.subscribe(obs_fail)
    event_emitter.subscribe(obs_after)
    # 例外が外に漏れないことも確認
    event_emitter.fire_event({}, {})

    return _assert(after == [1], "失敗 observer の後の observer も呼ばれる")


def test_unsubscribe():
    print("== unsubscribe: 解除後は呼ばれない ==")
    _setup()
    count = [0]

    def obs(state, entry):
        count[0] += 1

    event_emitter.subscribe(obs)
    event_emitter.fire_event({}, {})  # 1 回
    event_emitter.unsubscribe(obs)
    event_emitter.fire_event({}, {})  # 解除済、増えない

    return all([
        _assert(count[0] == 1, f"発火 1 回のみ (got {count[0]})"),
        _assert(len(event_emitter.list_subscribers()) == 0,
                "subscribers 空"),
    ])


def test_unsubscribe_unknown_no_error():
    print("== unsubscribe 未登録 observer: defensive で no-op ==")
    _setup()

    def obs(state, entry):
        pass

    # 例外出ないこと
    event_emitter.unsubscribe(obs)
    return _assert(True, "未登録 unsubscribe で例外なし")


def test_reset_subscribers():
    print("== reset_subscribers: 全 observer 解除 ==")
    _setup()
    event_emitter.subscribe(lambda s, e: None)
    event_emitter.subscribe(lambda s, e: None)
    event_emitter.reset_subscribers()
    return _assert(len(event_emitter.list_subscribers()) == 0,
                   "reset 後 subscribers 空")


def test_iteration_safe_during_subscribe():
    print("== fire_event 中の subscribe: snapshot で iteration 安全 ==")
    _setup()
    fired = []

    def obs1(state, entry):
        fired.append(1)
        # 発火中に新 observer 追加 (iteration を壊さないこと)
        event_emitter.subscribe(lambda s, e: fired.append(99))

    event_emitter.subscribe(obs1)
    event_emitter.fire_event({}, {})

    return all([
        _assert(fired == [1], "発火中の追加 observer は今回呼ばれない (snapshot)"),
        _assert(len(event_emitter.list_subscribers()) == 2,
                "次回発火用に追加されてる"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    print("test_event_emitter.py (段階13 Phase 0.2)")
    print("=" * 60)

    groups = [
        ("subscribe + fire_event 基本", test_subscribe_and_fire),
        ("複数 observer 順次発火", test_multiple_observers_in_order),
        ("重複 subscribe 無視", test_duplicate_subscribe_ignored),
        ("例外 isolation", test_exception_isolation),
        ("unsubscribe", test_unsubscribe),
        ("unsubscribe 未登録は no-op", test_unsubscribe_unknown_no_error),
        ("reset_subscribers", test_reset_subscribers),
        ("iteration safe (snapshot)", test_iteration_safe_during_subscribe),
    ]

    results = []
    for label, fn in groups:
        print()
        ok = fn()
        results.append((label, ok))

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  {passed}/{total} groups passed")

    # cleanup (他 test に影響しないよう)
    event_emitter.reset_subscribers()
    sys.exit(0 if passed == total else 1)
