"""世界モデル評価用デバッグログ (sandbox/wm_debug.jsonl) 解析スクリプト

使い方:
    python design/wm_analyze.py <profile_path>
    例: python design/wm_analyze.py ../test_wm_phase1

出力:
    - M1: channel_match_ratio (device_input pending 時のチャネル一致率)
    - M2: candidate_channel_distribution (LLM① 5候補のチャネル分布)
    - M3: response_tool_picked_by_channel (応答系ツール選択のチャネル別内訳)

段階1 合格ライン: M1 >= 0.80
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_events(profile_path: Path) -> list:
    log_file = profile_path / "sandbox" / "wm_debug.jsonl"
    if not log_file.exists():
        print(f"[error] {log_file} が存在しません。WM_DEBUG=1 で起動しましたか？", file=sys.stderr)
        sys.exit(1)
    events = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def analyze(events: list) -> dict:
    """イベント列から指標を計算して返す。"""
    results = {
        "total_events": len(events),
        "total_cycles": len({e["cycle"] for e in events if "cycle" in e}),
    }

    fires = [e for e in events if e["event"] == "fire"]
    candidates = [e for e in events if e["event"] == "candidates"]
    selections = [e for e in events if e["event"] == "selected"]

    # === M1: channel_match_ratio ===
    # device_input が pending にあるときの、応答系ツール選択の一致率
    # ch_match=None（internal）は分母から除外 → 応答ツールのみで評価
    device_pending_sel = [s for s in selections
                          if "device" in s.get("pending_channels", [])
                          and s.get("channel_match") is not None]
    if device_pending_sel:
        hits = sum(1 for s in device_pending_sel if s["channel_match"])
        m1_ratio = hits / len(device_pending_sel)
    else:
        m1_ratio = None

    results["M1_channel_match_ratio"] = {
        "value": m1_ratio,
        "hits": sum(1 for s in device_pending_sel if s.get("channel_match")),
        "total_evaluated": len(device_pending_sel),
        "note": "device_input pending 時、応答系ツールが device チャネルを選んだ割合",
        "pass_threshold": 0.80,
        "passed": m1_ratio is not None and m1_ratio >= 0.80,
    }

    # === M2: candidate_channel_distribution ===
    # LLM① が生成した候補のチャネル分布（偏向の可視化）
    cand_ch_counter = Counter()
    for c_event in candidates:
        for c in c_event.get("candidates", []):
            cand_ch_counter[c.get("channel", "?")] += 1
    total_cands = sum(cand_ch_counter.values())
    results["M2_candidate_channel_distribution"] = {
        "counts": dict(cand_ch_counter),
        "ratio": {ch: round(cnt / total_cands, 3) for ch, cnt in cand_ch_counter.items()} if total_cands else {},
        "total_candidates": total_cands,
    }

    # === M3: response_tool_picked_by_channel ===
    # 応答系ツール（internal 以外）の選択内訳
    sel_ch_counter = Counter()
    sel_tool_counter = Counter()
    for s in selections:
        ch = s.get("channel", "?")
        if ch != "internal":
            sel_ch_counter[ch] += 1
            sel_tool_counter[s["tool"]] += 1
    results["M3_response_tool_picked_by_channel"] = {
        "by_channel": dict(sel_ch_counter),
        "by_tool": dict(sel_tool_counter),
        "total_response_selections": sum(sel_ch_counter.values()),
    }

    # === 追加: 発火原因の分布 ===
    fire_causes = Counter(f.get("fire_cause", "?") for f in fires)
    results["fire_cause_distribution"] = dict(fire_causes)

    # === 追加: ミスマッチ事例 (デバッグ用) ===
    mismatches = []
    for s in selections:
        if s.get("channel_match") is False:
            mismatches.append({
                "cycle": s["cycle"],
                "tool": s["tool"],
                "selected_channel": s["channel"],
                "pending_channels": s["pending_channels"],
                "reason": s.get("reason", "")[:60],
            })
    results["mismatch_cases"] = mismatches[:10]  # 最大10件まで列挙
    results["mismatch_count_total"] = len([s for s in selections if s.get("channel_match") is False])

    return results


def print_report(results: dict):
    print("=" * 60)
    print(f"[World Model Debug Analysis]  サイクル数: {results['total_cycles']} / イベント数: {results['total_events']}")
    print("=" * 60)

    m1 = results["M1_channel_match_ratio"]
    print(f"\n[M1] channel_match_ratio (device pending 時の応答ツール一致率)")
    if m1["value"] is None:
        print(f"  → 評価対象イベントなし（device pending 時に応答ツール選択がなかった）")
    else:
        status = "✓ PASS" if m1["passed"] else "✗ FAIL"
        print(f"  {status}  {m1['value']:.2%}  ({m1['hits']}/{m1['total_evaluated']})  閾値: {m1['pass_threshold']:.0%}")

    m2 = results["M2_candidate_channel_distribution"]
    print(f"\n[M2] candidate_channel_distribution  (全候補 {m2['total_candidates']} 件)")
    for ch, ratio in sorted(m2["ratio"].items(), key=lambda x: -x[1]):
        bar = "█" * int(ratio * 40)
        print(f"  {ch:10s} {ratio:.1%}  {bar}")

    m3 = results["M3_response_tool_picked_by_channel"]
    print(f"\n[M3] response_tool_picked_by_channel  (応答選択 {m3['total_response_selections']} 回)")
    print("  チャネル別:")
    for ch, cnt in sorted(m3["by_channel"].items(), key=lambda x: -x[1]):
        print(f"    {ch:10s} {cnt:4d}")
    print("  ツール別 (top 10):")
    for tool, cnt in sorted(m3["by_tool"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {tool:25s} {cnt:4d}")

    print(f"\n[発火原因分布]")
    for cause, cnt in sorted(results["fire_cause_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {cause:15s} {cnt:4d}")

    mc = results["mismatch_count_total"]
    if mc > 0:
        print(f"\n[チャネル不一致事例] (全{mc}件中、最大10件)")
        for m in results["mismatch_cases"]:
            print(f"  cycle {m['cycle']:4d}: {m['tool']} ({m['selected_channel']}) "
                  f"← pending={m['pending_channels']}  理由: {m['reason']}")
    else:
        print(f"\n[チャネル不一致事例] なし")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    profile_path = Path(sys.argv[1]).resolve()
    if not profile_path.is_dir():
        print(f"[error] {profile_path} はディレクトリではありません", file=sys.stderr)
        sys.exit(1)
    events = load_events(profile_path)
    results = analyze(events)
    print_report(results)

    # JSON サマリも出す（後段で比較用）
    out_json = profile_path / "sandbox" / "wm_debug_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[summary saved] {out_json}")


if __name__ == "__main__":
    main()
