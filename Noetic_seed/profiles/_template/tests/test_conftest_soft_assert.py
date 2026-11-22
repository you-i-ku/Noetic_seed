"""PLAN v0.2: 隔離した子 pytest で採点・復元と検知経路の識別力を確認。"""
from pathlib import Path

import pytest


pytest_plugins = ["pytester"]
CONFTEST = Path(__file__).with_name("conftest.py")


@pytest.fixture
def isolated(pytester, monkeypatch):
    # 外側のプラグイン・警告設定を子プロセスへ持ち込まない。
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    pytester.makeini("[pytest]\nfilterwarnings = default")
    return pytester


def _run(pytester):
    return pytester.runpytest_subprocess("-q", "--tb=short", "-p", "no:cacheprovider")


def _expect_failures(result, count, *messages):
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=count, errors=0)
    output = result.stdout.str()
    for message in messages:
        assert message in output


@pytest.mark.parametrize("disabled", [None, "return", "assert"])
def test_independent_detection_and_mutations(isolated, disabled):
    source = CONFTEST.read_text(encoding="utf-8")
    replacements = {
        "return": ("returned_false = result is False", "returned_false = False"),
        "assert": ("failures.append(str(label))", "pass  # disabled recording"),
    }
    if disabled:
        old, new = replacements[disabled]
        assert source.count(old) == 1
        source = source.replace(old, new)
    isolated.makeconftest(source)
    isolated.makepyfile('''
        def _assert(cond, label):
            return cond

        def test_return_only():
            return False

        def test_assert_only():
            _assert(False, "assert-only-marker")
            return True
    ''')
    result = _run(isolated)
    if disabled is None:
        _expect_failures(result, 2, "test_return_only", "test_assert_only",
                         "test returned False", "_assert failed: assert-only-marker")
    else:
        # 正常実装用の同じ判定が、各変異で実際に失敗することを確認。
        with pytest.raises(AssertionError):
            _expect_failures(result, 2)
        result.assert_outcomes(passed=1, failed=1, errors=0)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        summary = result.stdout.str().split("short test summary info")[-1]
        expected = "test_assert_only" if disabled == "return" else "test_return_only"
        unexpected = "test_return_only" if disabled == "return" else "test_assert_only"
        assert expected in summary
        assert unexpected not in summary


def test_labels_keywords_and_original_behavior(isolated):
    isolated.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    isolated.makepyfile(test_labels='''
        def _assert(cond, label):
            print("original-output:" + label)
            return cond

        def test_mixed(capsys):
            assert _assert(1, "good") == 1
            assert _assert(0, "zero-label") == 0
            assert _assert(cond=[], label="empty-label") == []
            assert _assert(False, label="false-label") is False
            assert capsys.readouterr().out.splitlines() == [
                "original-output:good", "original-output:zero-label",
                "original-output:empty-label", "original-output:false-label"]
            return False
    ''', test_msg='''
        def _assert(cond, msg):
            return None

        def test_none_success():
            assert _assert(cond=True, msg="success-none") is None

        def test_none_failure():
            assert _assert(cond=False, msg="failure-none") is None
    ''')
    result = _run(isolated)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, failed=2, errors=0)
    output = result.stdout.str()
    assert "test returned False; _assert failed: zero-label, empty-label, false-label" in output
    assert "_assert failed: failure-none" in output
    assert "_assert failed: success-none" not in output


def test_fixtures_parametrize_single_call_and_restoration(isolated):
    isolated.makeconftest(CONFTEST.read_text(encoding="utf-8") + '''

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    original_assert = item.module._assert
    original_test = item.obj
    try:
        return (yield)
    finally:
        assert item.module._assert is original_assert
        assert item.obj is original_test
''')
    isolated.makepyfile('''
        import pytest
        calls = []

        def _assert(cond, label):
            return cond

        original_assert = _assert

        @pytest.fixture(autouse=True)
        def unrelated():
            return "not a test argument"

        @pytest.fixture
        def value():
            return 3

        @pytest.mark.parametrize("number", [0, 1])
        def test_once(value, number):
            assert value == 3
            assert _assert.__wrapped__ is original_assert
            calls.append(number)
            assert calls == list(range(number + 1))
            _assert(number == 1, "first-only")

        def test_after():
            assert calls == [0, 1]
            assert _assert.__wrapped__ is original_assert
            _assert(True, "clean-next-test")
    ''')
    result = _run(isolated)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=2, failed=1, errors=0)
    assert "_assert failed: first-only" in result.stdout.str()


def test_existing_exceptions_skip_xfail_and_restoration(isolated):
    isolated.makeconftest(CONFTEST.read_text(encoding="utf-8") + '''

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    original_assert, original_test = item.module._assert, item.obj
    try:
        return (yield)
    finally:
        assert item.module._assert is original_assert
        assert item.obj is original_test
''')
    isolated.makepyfile('''
        import pytest

        def _assert(cond, msg):
            if msg == "original-error":
                raise ValueError(msg)
            return cond

        def test_exception():
            _assert(False, "soft-before-exception")
            raise RuntimeError("original-runtime")

        def test_assert_exception():
            _assert(False, "original-error")

        def test_skip():
            _assert(False, "soft-before-skip")
            pytest.skip("original-skip")

        def test_xfail():
            _assert(False, "soft-before-xfail")
            pytest.xfail("original-xfail")

        @pytest.mark.xfail(strict=True, reason="expected soft failure")
        def test_marked_xfail():
            _assert(False, "marked")

        def test_after():
            _assert(True, "clean")
    ''')
    result = _run(isolated)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, failed=2, skipped=1, xfailed=2, errors=0)
    output = result.stdout.str()
    assert "RuntimeError: original-runtime" in output
    assert "ValueError: original-error" in output
    assert "_assert failed:" not in output


def test_no_assert_and_non_false_returns(isolated):
    isolated.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    isolated.makepyfile('''
        import pytest

        @pytest.mark.parametrize("value", [None, True, 0, []])
        def test_return(value):
            return value
    ''')
    result = _run(isolated)
    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=4, warnings=3, errors=0)
    assert "PytestReturnNotNoneWarning" in result.stdout.str()


def test_async_is_left_to_pytest(isolated):
    isolated.makepyfile('''
        async def test_coroutine():
            return False

        async def test_async_generator():
            yield False

        class Awaitable:
            def __await__(self):
                yield

        class AsyncIterator:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        def test_returns_awaitable():
            return Awaitable()

        def test_returns_async_iterator():
            return AsyncIterator()
    ''')
    baseline = _run(isolated)
    isolated.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    result = _run(isolated)
    assert result.ret == baseline.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(failed=4, errors=0)
    assert result.parseoutcomes() == baseline.parseoutcomes()
    assert result.stdout.str().count("async def functions are not natively supported") == 4
    assert "test returned False" not in result.stdout.str()
    assert "_assert failed:" not in result.stdout.str()
