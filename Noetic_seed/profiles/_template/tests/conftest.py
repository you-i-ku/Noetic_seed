"""既存テストの False 返却と soft assert を pytest の失敗にする。"""
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction, signature

import pytest


_SOFT_FAILURES = pytest.StashKey[list]()


@pytest.fixture(autouse=True)
def _record_soft_asserts(request):
    failures = []
    request.node.stash[_SOFT_FAILURES] = failures
    module = request.module
    original = getattr(module, "_assert", None)
    if not callable(original):
        yield
        return

    parameters = signature(original)
    condition_name = next(iter(parameters.parameters))

    @wraps(original)
    def recorded(*args, **kwargs):
        # 元の表示・返り値・例外を保つ。成功時 None の実装もある。
        result = original(*args, **kwargs)
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments[condition_name]:
            label = bound.arguments.get("label", bound.arguments.get("msg"))
            failures.append(str(label))
        return result

    module._assert = recorded
    try:
        yield
    finally:
        module._assert = original


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    if not isinstance(item, pytest.Function):
        return (yield)

    failures = item.stash.get(_SOFT_FAILURES, [])
    failures.clear()  # setup 中の呼び出しはテスト本体の採点に含めない。
    original = item.obj
    returned_false = False

    @wraps(original)
    def checked(*args, **kwargs):
        nonlocal returned_false
        result = original(*args, **kwargs)
        returned_false = result is False
        return result

    # 非同期関数の識別と awaitable / async iterator の処理は pytest に任せる。
    if not (iscoroutinefunction(original) or isasyncgenfunction(original)):
        item.obj = checked
    try:
        result = yield
    finally:
        item.obj = original

    # yield が例外・skip・xfail を送出した場合はここへ到達しない。
    reasons = []
    if returned_false:
        reasons.append("test returned False")
    if failures:
        reasons.append("_assert failed: " + ", ".join(failures))
    if reasons:
        pytest.fail("; ".join(reasons), pytrace=False)
    return result
