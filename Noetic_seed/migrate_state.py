"""state.json 互換性マイグレーションスクリプト。

旧コード版で動作していた profile の state.json を新コード版に移行する。
互換性修正は安全（情報損失なし）、--trim は鮮度勾配に合わせて破壊的に log を圧縮する。

使用例:
  # 安全モード（互換性修正のみ）
  python Noetic_seed/migrate_state.py Noetic_seed/ikuset/state.json

  # 圧縮モード（鮮度勾配で古いログを永続切詰）
  python Noetic_seed/migrate_state.py Noetic_seed/ikuset/state.json --trim

スクリプトは実行前に state.json.bak を作成する（元に戻せる）。
"""
import json
import sys
import shutil
from pathlib import Path

# 鮮度勾配（prompt.py の log_gradient と一致させる）
GRADIENT_BOUNDARIES = [5, 15, 45]
GRADIENT_CAPS = [20000, 3000, 800, 200]


def _tier_cap(pos_from_end: int) -> int:
    """鮮度位置から result cap を返す。"""
    for i, b in enumerate(GRADIENT_BOUNDARIES):
        if pos_from_end < b:
            return GRADIENT_CAPS[i]
    return GRADIENT_CAPS[-1]


def migrate(state_path_str: str, trim: bool = False) -> int:
    state_path = Path(state_path_str).resolve()
    if not state_path.exists():
        print(f"ERROR: {state_path} が存在しません")
        return 1

    # バックアップ
    backup_path = state_path.with_name(state_path.stem + ".bak.json")
    shutil.copy(state_path, backup_path)
    print(f"[backup] {backup_path}")

    size_before = state_path.stat().st_size

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: {state_path} の JSON パース失敗: {e}")
        return 1

    changes = []

    # --- ① state['plan'] の削除 ---
    if "plan" in state:
        del state["plan"]
        changes.append("state['plan'] を削除")

    # --- ② pending の user_message → external_message ---
    pending = state.get("pending", [])
    renamed_pending = 0
    for p in pending:
        if p.get("type") == "user_message":
            p["type"] = "external_message"
            renamed_pending += 1
    if renamed_pending:
        changes.append(f"pending {renamed_pending}件: user_message → external_message")

    # --- ③ log エントリの result プレフィックス 'user: ' → 'external: ' ---
    log = state.get("log", [])
    renamed_log = 0
    for entry in log:
        result = entry.get("result", "")
        if isinstance(result, str) and result.startswith("user: "):
            entry["result"] = "external: " + result[len("user: "):]
            renamed_log += 1
    if renamed_log:
        changes.append(f"log {renamed_log}件: 'user: ' → 'external: '")

    # --- ④ オプション: 鮮度勾配で log を圧縮（破壊的）---
    if trim:
        n = len(log)
        chars_before = 0
        chars_after = 0
        trimmed_count = 0
        for i, entry in enumerate(log):
            pos = n - 1 - i
            cap = _tier_cap(pos)
            result = entry.get("result", "")
            if isinstance(result, str):
                chars_before += len(result)
                if len(result) > cap:
                    orig_len = len(result)
                    entry["result"] = result[:cap] + f"\n[migrated: 表示上 {cap}/{orig_len}字に圧縮]"
                    trimmed_count += 1
                chars_after += len(entry.get("result", ""))

        if chars_before > 0 and trimmed_count > 0:
            reduction_pct = round((1 - chars_after / chars_before) * 100)
            changes.append(
                f"log {trimmed_count}件を鮮度勾配で圧縮: {chars_before:,}字 → {chars_after:,}字 (-{reduction_pct}%)"
            )

    # --- 保存 ---
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_after = state_path.stat().st_size

    # --- レポート ---
    print()
    print("=" * 50)
    print(f"Migration complete: {state_path.name}")
    print("=" * 50)
    if not changes:
        print("変更なし（既に新形式）")
    else:
        for c in changes:
            print(f"  ✓ {c}")
    print()
    size_reduction = size_before - size_after
    if size_reduction > 0:
        pct = round(size_reduction / size_before * 100)
        print(f"ファイルサイズ: {size_before:,} → {size_after:,} bytes (-{pct}%)")
    else:
        print(f"ファイルサイズ: {size_before:,} → {size_after:,} bytes")
    print(f"バックアップ: {backup_path}")
    print()
    print("元に戻す:  cp {} {}".format(backup_path.name, state_path.name))

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python migrate_state.py <state.json path> [--trim]")
        print()
        print("  <state.json path>  対象プロファイルの state.json パス")
        print("  --trim             鮮度勾配で log を永続圧縮（破壊的、情報損失）")
        print()
        print("  互換性修正（plan削除、user_message→external_message、user:→external:）は常に実施")
        print("  事前に state.bak.json としてバックアップが作成される")
        sys.exit(1)

    path = sys.argv[1]
    trim = "--trim" in sys.argv
    sys.exit(migrate(path, trim=trim))
