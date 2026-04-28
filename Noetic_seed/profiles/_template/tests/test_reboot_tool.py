"""tools/reboot.py の unit test (段階12 Step 4, PLAN §8-2 / §13-1)。

reboot は os._exit / subprocess.Popen / stop_ws_server を実呼出するため、
全 7 経路を mock して経路順序と短絡を検証する。実 process は spawn しない。

検証ケース:
  - 承認 accept: 5 段階 (approval, save_state, stop_ws_server, sleep, Popen, _exit)
    が想定順序で呼ばれ、_exit(0) で終わる
  - 承認 reject: キャンセル message 返り、後続 6 段階は呼ばれない (短絡)
  - args message なし: preview に [reboot] prefix と '再起動' 文言が入る

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_reboot_tool.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


@patch("tools.reboot.os._exit")
@patch("tools.reboot.time.sleep")
@patch("tools.reboot.subprocess.Popen")
@patch("tools.reboot.stop_ws_server")
@patch("tools.reboot.load_state", return_value={})
@patch("tools.reboot.save_state")
@patch("tools.reboot.request_approval", return_value=True)
def test_reboot_accept_full_sequence(
    mock_approval, mock_save, mock_load, mock_stop_ws,
    mock_popen, mock_sleep, mock_exit,
):
    """承認 accept で 5 段階全部走り、最後 _exit(0)。"""
    print("== reboot 承認 accept で全段階呼出 + os._exit(0) ==")
    from tools.reboot import _reboot
    _reboot({"message": "段階12 Step 1 反映"})
    return all([
        _assert(mock_approval.called, "request_approval 呼ばれた"),
        _assert("[reboot]" in mock_approval.call_args[0][1],
                "preview に [reboot] prefix"),
        _assert("段階12 Step 1 反映" in mock_approval.call_args[0][1],
                "preview に message 挿入"),
        _assert(mock_save.called, "save_state 呼ばれた"),
        _assert(mock_load.called, "load_state 呼ばれた"),
        _assert(mock_stop_ws.called, "stop_ws_server 呼ばれた"),
        _assert(mock_sleep.called, "time.sleep 呼ばれた (port handoff)"),
        _assert(mock_popen.called, "subprocess.Popen 呼ばれた"),
        _assert(mock_exit.called, "os._exit 呼ばれた"),
        _assert(mock_exit.call_args[0][0] == 0, "exit code = 0"),
    ])


@patch("tools.reboot.os._exit")
@patch("tools.reboot.subprocess.Popen")
@patch("tools.reboot.stop_ws_server")
@patch("tools.reboot.save_state")
@patch("tools.reboot.request_approval", return_value=False)
def test_reboot_reject_short_circuit(
    mock_approval, mock_save, mock_stop_ws, mock_popen, mock_exit,
):
    """承認 reject でキャンセル message 返り、後続段階は呼ばれない。"""
    print("== reboot 承認 reject で短絡、後続段階 skip ==")
    from tools.reboot import _reboot
    result = _reboot({})
    return all([
        _assert(mock_approval.called, "request_approval 呼ばれた"),
        _assert("キャンセル" in result, f"返り値にキャンセル message (実測: {result})"),
        _assert(not mock_save.called, "save_state 呼ばれない (短絡)"),
        _assert(not mock_stop_ws.called, "stop_ws_server 呼ばれない (短絡)"),
        _assert(not mock_popen.called, "subprocess.Popen 呼ばれない (短絡)"),
        _assert(not mock_exit.called, "os._exit 呼ばれない (短絡)"),
    ])


@patch("tools.reboot.request_approval", return_value=False)
def test_reboot_preview_without_message(mock_approval):
    """args に message なしでも preview は最低限の説明を含む。"""
    print("== message なしでも preview 最低限の文言 ==")
    from tools.reboot import _reboot
    _reboot({})
    preview = mock_approval.call_args[0][1]
    return all([
        _assert("[reboot]" in preview, "preview に [reboot] prefix"),
        _assert("再起動" in preview, "preview に '再起動' 文言"),
    ])


if __name__ == "__main__":
    groups = [
        ("reboot 承認 accept で全段階呼出", test_reboot_accept_full_sequence),
        ("reboot 承認 reject で短絡", test_reboot_reject_short_circuit),
        ("preview message なしでも最低限", test_reboot_preview_without_message),
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
