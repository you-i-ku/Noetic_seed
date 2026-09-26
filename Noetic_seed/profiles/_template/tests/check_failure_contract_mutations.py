"""Run isolated in-memory mutants. Exit 0 only when every mutant is killed.

Usage: python -B tests/check_failure_contract_mutations.py
Production files are never edited; no provider/network/process tools are called.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_failure_contract_foundation.py::"
MUTANTS = [('ToolError prefix',
  'core.runtime.conversation',
  'str(e) if expected else',
  'f"tool execution error: {e}" if expected else',
  'test_runtime_contract_and_hooks'),
 ('failure uses success hook',
  'core.runtime.conversation',
  'post = self.hook_runner.run_post_tool_use_failure(',
  'post = self.hook_runner.run_post_tool_use(',
  'test_runtime_contract_and_hooks'),
 ('callback drops detail',
  'core.runtime.conversation',
  '            return rec\n        return executor',
  '            rec.detail = None\n            return rec\n        return executor',
  'test_claude_sdk_callback_and_round_trip'),
 ('wrong SDK key',
  'core.providers.claude_code',
  '"is_error": rec.is_error,',
  '"isError": rec.is_error,',
  'test_claude_sdk_callback_and_round_trip'),
 ('unredacted message',
  'core.runtime.conversation',
  'rec.output = redact(output)',
  'rec.output = str(output)',
  'test_secret_removal_before_hooks_providers_and_observations'),
 ('accept old history',
  'core.metrics',
  '    if version != required:',
  '    if False:',
  'test_version_validation_live_rebuild_summary_and_stage3_gate'),
 ('main drops detail',
  'main',
  '"detail": copy.deepcopy(_r.detail)',
  '"detail": None',
  'test_main_observation_copy_is_structured'),
 ('invocation omits version',
  'core.metrics',
  '            "failure_contract": ACTIVE_FAILURE_CONTRACT,',
  '',
  'test_stage3_writers_and_default_readers'),
 ('prediction omits version',
  'core.metrics',
  '                       "failure_contract": self.required_contract if self.required_contract is not '
  'None else ACTIVE_FAILURE_CONTRACT,',
  '',
  'test_stage3_writers_and_default_readers'),
 ('default contract stays v1',
  'core.metrics',
  'ACTIVE_FAILURE_CONTRACT = 2',
  'ACTIVE_FAILURE_CONTRACT = 1',
  'test_stage3_writers_and_default_readers'),
 ('rebuild mixes v1',
  'core.metrics',
  '_, reason = self._accept(event)',
  '_, reason = self._accept({**event, "failure_contract": 2})',
  'test_stage3_writers_and_default_readers'),
 ('metrics retains prefix fallback',
  'core.metrics',
  '    if not is_error:\n        return "ok"',
  '    if not is_error:\n'
  '        return "tool_error" if str(output).lstrip().startswith(("Error", "エラー")) else "ok"',
  'test_stage3_status_uses_structure_only'),
 ('main retains prefix counters',
  'main',
  'if _p and not rec.is_error:',
  'if _p and not str(rec.output).startswith(("Error", "エラー", "該当なし")):',
  'test_stage3_main_counts_by_is_error'),
 ('failed files increment counts',
  'main',
  'if _p and not rec.is_error:',
  'if _p:',
  'test_stage3_main_counts_by_is_error'),
 ('failed memory increments count',
  'main',
  'if not rec.is_error:',
  'if True:',
  'test_stage3_main_counts_by_is_error'),
 ('learning retains rejected prefix',
  'main',
  '_ok = not invocation.is_error',
  '_ok = not invocation.is_error and not str(invocation.output).startswith("[REJECTED]")',
  'test_stage3_success_rejected_text_keeps_learning'),
 ('secret_read failure bypasses redaction',
  'core.runtime.conversation',
  'if rec.tool_name == "secret_read" and not is_error else None',
  'if rec.tool_name == "secret_read" else None',
  'test_secret_read_exception_is_success_only'),
 ('prompt trace leaks provider body',
  'core.prompt_trace',
  'request.observation_messages if request.observation_messages is not None else request.messages',
  'request.messages',
  'test_secret_read_success_provider_original_observation_redacted'),
 ('main memory status drops error kind',
  'main',
  'tool_execution_status(str(invocation.output), invocation.is_error, invocation.error_kind)',
  'tool_execution_status(str(invocation.output), invocation.is_error)',
  'test_stage3_main_counts_by_is_error'),
 ('secret observation retains bare body',
  'core.runtime.conversation',
  'if rec.provider_output is not None:',
  'if False:',
  'test_secret_read_success_provider_original_observation_redacted'),
 ('memory console leaks exception',
  'core.memory',
  'result_redactor()(str(e))',
  'str(e)',
  'test_memory_lower_exception_console_redacted'),
 ('link console leaks exception',
  'core.memory_links',
  'result_redactor()(str(e))',
  'str(e)',
  'test_memory_lower_exception_console_redacted'),
 ('http truncates before redaction',
  'tools.http_tool',
  'redact(url)[:80]',
  'redact(url[:80])',
  'test_http_redacts_before_truncation'),
 ('display log truncates before redaction',
  'tools.ui_tools',
  'redact(content)[:100]',
  'redact(content[:100])',
  'test_display_paths_redact_before_shortening'),
 ('web body truncates before redaction',
  'core.runtime.tools.web',
  'body = result_redactor()(resp.text)',
  'body = resp.text',
  'test_display_paths_redact_before_shortening'),
 ('two state writers',
  'core.state',
  '    if live is not None:\n        return live',
  '    if False:\n        return live',
  'test_live_tool_failure_then_success_and_cycle_save'),
 ('main replacement clears live',
  'main',
  '        result = _base_post_hook(tool_name, tool_input, output)',
  '        fresh = load_state()\n'
  '        state.clear()\n'
  '        state.update(fresh)\n'
  '        result = _base_post_hook(tool_name, tool_input, output)',
  'test_live_tool_failure_then_success_and_cycle_save'),
 ('generation copied after save',
  'core.state',
  '            _retain_generation()',
  '            _atomic_write(STATE_FILE, json.dumps(state))\n            _retain_generation()',
  'test_generations_saved_before_write_and_capped'),
 ('recovery keeps stale views',
  'core.state',
  '    _rebuild_views_from_jsonl(data)\n',
  '    pass\n',
  'test_startup_recovery_latest_valid_views_cycles_and_records'),
 ('reboot exits on save failure',
  'tools.reboot',
  '    if not save_state(load_state()):',
  '    if save_state(load_state()) and False:',
  'test_reboot_requires_successful_save'),
 ('tools omit shared lock',
  'core.runtime.conversation',
  '    @state_locked\n    def _execute_tool_use',
  '    def _execute_tool_use',
  'test_tool_and_save_share_lock_across_threads'),
 ('recovery rolls cycle backwards',
  'core.state',
  '    data["cycle_id"] = latest',
  '    pass',
  'test_startup_recovery_latest_valid_views_cycles_and_records'),
 ('sanity blocks recovery',
  'main',
  'profile_name=BASE_DIR.name, recover_state=True',
  'profile_name=BASE_DIR.name, recover_state=False',
  'test_main_startup_sanity_allows_state_recovery'),
 ('approval skips recheck',
  'tools.device_tools',
  '    if state.get("stream_active"):\n'
  '        raise ToolError("stream became active while awaiting approval")',
  '    if False:\n        raise ToolError("stream became active while awaiting approval")',
  'test_device_rechecks_state_after_approval'),
 ('sanity check reloads live modules',
  'core.sanity_check',
  '    script = (',
  '    for name in ("core.controller", "tools", *_enumerate_runtime_modules()):\n'
  '        if name in sys.modules:\n'
  '            importlib.reload(sys.modules[name])\n'
  '        else:\n'
  '            importlib.import_module(name)\n'
  '    script = (',
  'test_sanity_check_preserves_tool_error_identity')]


MUTANTS.extend([
    ("search error leaks protected path", "core.runtime.tools.file_ops",
     'f"Error: search failed ({kind})"', 'f"Error: search failed ({exc})"',
     'tests/test_failure_contract_file_shell_web.py::test_search_path_errors_never_expose_secret_names'),
    ("partial reflect rejected", "core.reflection",
     '                skipped[kind] = skipped.get(kind, 0) + 1',
     '                raise ToolError("bad line rejects entire response")',
     'test_manual_reflect_partial_success_and_detail'),
    ("readable reflect items dropped", "core.reflection",
     '            accepted.append(line)', '            pass',
     'test_manual_reflect_partial_success_and_detail'),
    ("empty reflect is success", "core.reflection",
     '    if not readable:', '    if False:',
     'test_manual_reflect_empty_sections_fail_and_automatic_fallback'),
    ("manual LLM failure swallowed", "core.reflection",
     '        if strict:\n            if isinstance(e, ToolError):',
     '        if False:\n            if isinstance(e, ToolError):',
     'test_manual_reflect_failure_not_zero_success'),
    ("reflect detail dropped", "main",
     'detail=result.get("parse_detail")', 'detail=None',
     'test_manual_reflect_partial_success_and_detail'),
    ("automatic reflect becomes strict", "core.reflection",
     'else reflect_fn(state, call_llm_fn)', 'else reflect_fn(state, call_llm_fn, strict=True)',
     'test_automatic_reflect_keeps_permissive_behavior'),
    ("root search rejected again", "core.runtime.hooks",
     '        # Root search is allowed;',
     '        if tool_name in ("glob_search", "grep_search") and rel == ".":\n'
     '            return HookRunResult.deny(["root scan denied"])\n        # Root search is allowed;',
     'tests/test_failure_contract_file_shell_web.py::test_root_search_filters_secret_paths_before_reads'),
    ("protected search candidates leak", "core.runtime.tools.file_ops",
     '            excluded += 1\n            continue', '            excluded += 1',
     'tests/test_failure_contract_file_shell_web.py::test_root_search_filters_secret_paths_before_reads'),
    ("grep bypasses candidate filtering", "core.runtime.tools.file_ops",
     'candidates, excluded = _search_candidates(workspace_root, start, glob_pat)',
     'candidates, excluded = sorted(start.glob(glob_pat)), 0',
     'tests/test_failure_contract_file_shell_web.py::test_root_search_filters_secret_paths_before_reads'),
    ("public search candidates dropped", "core.runtime.tools.file_ops",
     '        candidates.append(path)', '        pass',
     'tests/test_failure_contract_file_shell_web.py::test_root_search_filters_secret_paths_before_reads'),
    ("brace glob silently empty", "core.runtime.tools.file_ops",
     '    if "{" in pattern or "}" in pattern:', '    if False:',
     'tests/test_failure_contract_file_shell_web.py::test_glob_braces_are_explicit_tool_failure'),
    ("protected aliases leak", "core.runtime.tools.file_ops",
     'or target == secret_file.resolve() or target.is_relative_to(secret_dir.resolve())',
     'or False',
     'tests/test_failure_contract_file_shell_web.py::test_search_aliases_and_outside_boundary'),
])


def main():
    killed = 0
    for label, module, before, after, test in MUTANTS:
        if module == "main":
            setup = f'''
from pathlib import Path
original = Path.read_text
def read(self, *args, **kwargs):
    source = original(self, *args, **kwargs)
    if self.name == "main.py" and self.parent == Path.cwd():
        assert {before!r} in source
        return source.replace({before!r}, {after!r})
    return source
Path.read_text = read
'''
        else:
            setup = f'''
import importlib
from pathlib import Path
module = importlib.import_module({module!r})
source = Path(module.__file__).read_text(encoding="utf-8")
assert {before!r} in source
exec(compile(source.replace({before!r}, {after!r}), module.__file__, "exec"), module.__dict__)
'''
        target = test if test.startswith("tests/") else TEST + test
        command = setup + f'''
import pytest
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--disable-warnings", {target!r}]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", command], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        detected = result.returncode == 1 and "failed" in result.stdout
        killed += detected
        print(f"{label}: {'KILLED' if detected else 'NOT VERIFIED'}")
        if not detected:
            print(result.stdout[-2000:], result.stderr[-2000:])
    print(f"{killed}/{len(MUTANTS)} mutants killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
