"""本体と観察で共有する成否判定。観察の組み立て処理には依存しない。"""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import metrics


@pytest.mark.parametrize("output,is_error,expected", [
    ("[REJECTED] denied", False, "rejected"),
    ("[REJECTED] denied", True, "rejected"),
    ("failure", True, "runtime_error"),
    ("Error: failed", True, "runtime_error"),
    ("エラー: failed", False, "tool_error"),
    ("Error: failed", False, "tool_error"),
    (" \nエラー: failed", False, "tool_error"),
    (" \tError: failed", False, "tool_error"),
    (" [REJECTED] denied", False, "ok"),
    ("error: lowercase", False, "ok"),
    ("結果の途中に Error", False, "ok"),
    ("", False, "ok"),
    ("成功", False, "ok"),
])
def test_tool_execution_status(output, is_error, expected, monkeypatch):
    """優先順・空白・大文字小文字の変更、観察関数への逆依存を検出する。"""
    observer = Mock(side_effect=RuntimeError("observer failed"))
    monkeypatch.setattr(metrics, "build_outward_execution", observer)
    assert metrics.tool_execution_status(output, is_error) == expected
    observer.assert_not_called()
