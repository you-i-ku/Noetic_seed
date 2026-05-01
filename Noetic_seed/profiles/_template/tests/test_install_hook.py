"""make_install_command_check_hook の test (段階12 Step 7.5 ③, PLAN §3-5-2 ③)。

検証ケース:
  - 存在する pkg (requests) → allow、warning なし
  - 存在しない pkg (asdfqwerty_nonexistent_xyz) → deny
  - typosquatting (requets ≒ requests、Levenshtein 1) → allow + warning
  - 非 install command (pip list / pip search 等) → passthrough
  - 非 pip command (npm install 等) → passthrough (本実装は pip only MVP)
  - ネットワーク不通 (http_get None 返り) → allow + warning (PLAN §3-5-2 ③ fallback)
  - cache 効果 (同 pkg 連続呼出で fetch 1 回のみ)
  - python -m pip install パターン

実 PyPI に問合せず、http_get inject で mock。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_install_hook.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import make_install_command_check_hook


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_mock_http_get(known_pkgs: dict, network_down: bool = False):
    """known_pkgs: {pkg_name: pkg_data_dict} を返す mock factory。
    network_down=True なら全問合せに None 返り (不通シミュレート)。
    list `calls` で問合せ URL を記録する。"""
    calls = []

    def fetch(url):
        calls.append(url)
        if network_down:
            return None
        for pkg_name, data in known_pkgs.items():
            if f"/{pkg_name}/json" in url:
                return json.dumps({"info": {"name": pkg_name}, "releases": {}})
        return "__NOT_FOUND__"
    return fetch, calls


def test_existing_pkg_allowed():
    print("== pip install requests (存在 pkg) は allow ==")
    fetch, _calls = _make_mock_http_get({"requests": {}})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install requests"})
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(not r.messages, f"warnings なし (実: {r.messages})"),
    ])


def test_nonexistent_pkg_denied():
    print("== pip install <hallucinated> (存在しない pkg) は deny ==")
    fetch, _calls = _make_mock_http_get({})  # known_pkgs 空 = 全部 not found
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install asdfqwerty_nonexistent_xyz"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("hallucination" in m for m in r.messages),
                "deny reason に 'hallucination' 含む"),
    ])


def test_typosquatting_warned():
    print("== pip install requets (≒ requests, Levenshtein 1) は warning ==")
    fetch, _calls = _make_mock_http_get({"requets": {}})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install requets"})
    return all([
        _assert(not r.denied, "denied=False (typosquat は警告のみ)"),
        _assert(any("Levenshtein" in m and "requests" in m for m in r.messages),
                "warning に Levenshtein + 人気 pkg 名 含む"),
    ])


def test_pip_list_not_install_passthrough():
    print("== pip list は install じゃないので passthrough ==")
    fetch, calls = _make_mock_http_get({})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip list"})
    return all([
        _assert(not r.denied, "allow"),
        _assert(len(calls) == 0, "PyPI 問合せなし (passthrough)"),
    ])


def test_npm_install_passthrough():
    print("== npm install は対象外 (本実装は pip only MVP) ==")
    fetch, calls = _make_mock_http_get({})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "npm install lodash"})
    return all([
        _assert(not r.denied, "allow"),
        _assert(len(calls) == 0, "PyPI 問合せなし (npm は対象外)"),
    ])


def test_network_down_warning_only():
    print("== ネットワーク不通は warning + allow (fallback) ==")
    fetch, _calls = _make_mock_http_get({}, network_down=True)
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install requests"})
    return all([
        _assert(not r.denied, "denied=False (PLAN §3-5-2 ③ fallback)"),
        _assert(any("ネットワーク不通" in m for m in r.messages),
                "warning に 'ネットワーク不通' 含む"),
    ])


def test_cache_effect():
    print("== 連続呼出で cache が効く (fetch 1 回のみ) ==")
    fetch, calls = _make_mock_http_get({"requests": {}})
    hook = make_install_command_check_hook(http_get=fetch)
    hook("bash", {"command": "pip install requests"})
    hook("bash", {"command": "pip install requests"})
    hook("bash", {"command": "pip install requests"})
    return _assert(len(calls) == 1,
                   f"PyPI 問合せ 1 回 (cache hit、実測: {len(calls)})")


def test_python_m_pip_install_pattern():
    print("== python -m pip install requests も検出 ==")
    fetch, _calls = _make_mock_http_get({"requests": {}})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "python -m pip install requests"})
    return all([
        _assert(not r.denied, "denied=False (存在 pkg)"),
        _assert(len(_calls) == 1, "PyPI 問合せ 1 回 (パターン検出 OK)"),
    ])


def test_version_specifier_extracted():
    print("== pip install requests==2.0 で pkg 名のみ抽出 ==")
    fetch, _calls = _make_mock_http_get({"requests": {}})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install requests==2.31.0"})
    return all([
        _assert(not r.denied, "denied=False"),
        _assert(any("/requests/json" in c for c in _calls),
                "URL に '/requests/json' (version 部分は除外)"),
    ])


def test_local_path_install_skipped():
    print("== pip install -e . (local path install) は対象外 ==")
    fetch, calls = _make_mock_http_get({})
    hook = make_install_command_check_hook(http_get=fetch)
    r = hook("bash", {"command": "pip install -e ."})
    return all([
        _assert(not r.denied, "allow"),
        _assert(len(calls) == 0, "PyPI 問合せなし (local install は対象外)"),
    ])


if __name__ == "__main__":
    groups = [
        ("存在 pkg は allow", test_existing_pkg_allowed),
        ("存在しない pkg は deny", test_nonexistent_pkg_denied),
        ("typosquatting は warning", test_typosquatting_warned),
        ("pip list は passthrough", test_pip_list_not_install_passthrough),
        ("npm install は対象外", test_npm_install_passthrough),
        ("ネットワーク不通は warning + allow", test_network_down_warning_only),
        ("cache 効果", test_cache_effect),
        ("python -m pip install 検出", test_python_m_pip_install_pattern),
        ("version specifier 抽出", test_version_specifier_extracted),
        ("local path install は対象外", test_local_path_install_skipped),
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
