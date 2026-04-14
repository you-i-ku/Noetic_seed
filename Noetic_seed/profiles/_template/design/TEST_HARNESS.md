# 世界モデル評価用テストハーネス

WORLD_MODEL.md 段階1〜6 の効果を定量評価するためのテスト手順と計測基盤。

---

## 1. 設計方針

- **記憶非依存**: 空プロファイル（既存記憶なし）からスタート可能
- **段階横断**: 段階1 のチャネル一致率だけでなく、段階5 の予測精度まで同じ基盤で測れる
- **A/B 不要**: 絶対指標（例: channel_match_ratio ≥ 0.80）で合否判定
- **実運用無影響**: `WM_DEBUG` 環境変数で on/off、未設定時はゼロコスト

---

## 2. 構成要素

### 2.1 計測ポイント (main.py 内)

```python
_WM_DEBUG = os.environ.get("WM_DEBUG") == "1"
_WM_LOG_PATH = BASE_DIR / "sandbox" / "wm_debug.jsonl"

def _wm_log(event_type, payload): ...
```

3 つの event を `sandbox/wm_debug.jsonl` に追記:

| event | 発生タイミング | 主要フィールド |
|-------|------------|-------------|
| `fire` | サイクル発火時 | cycle, fire_cause, pressure, pending_channels, pending_count |
| `candidates` | LLM① 候補生成後 | cycle, candidates[{tool, channel, reason}] |
| `selected` | controller 選択後 | cycle, tool, channel, pending_channels, channel_match |

`channel_match` のルール:
- `None`: 選択ツールが internal（応答系でない）または pending に channel なし
- `True`: 選択ツールの channel が pending channel に含まれる
- `False`: 選択ツールは応答系だが pending と違うチャネル ← **これがバグ**

### 2.2 解析スクリプト

`design/wm_analyze.py` — jsonl を読んで指標を計算。

```bash
python design/wm_analyze.py <profile_path>
```

出力:
- **M1** `channel_match_ratio`: device pending 時の応答ツール一致率
- **M2** `candidate_channel_distribution`: 候補チャネル分布
- **M3** `response_tool_picked_by_channel`: 応答選択のチャネル別内訳
- 発火原因分布、ミスマッチ事例（最大10件）
- `wm_debug_summary.json` も出力（後段比較用）

---

## 3. 段階別の合格基準

### 段階1: チャネルタグ修正

| 指標 | 合格 |
|-----|-----|
| M1 (channel_match_ratio) | ≥ 0.80 |
| M2 で device pending 時に device チャネル候補が 1 件以上含まれる | ほぼ毎回 |
| mismatch_count_total が 50サイクルあたり 3件以下 | ◎ |

### 段階2-3: 世界モデル基盤+更新

追加指標（段階2 以降 WM_DEBUG で出力）:
- `entity_created_count`: 新 entity 登録回数
- `fact_updated_count`: fact 更新回数
- `confidence_avg`: fact 信頼度の平均（β+ が機能している指標）

### 段階5: Predictor

追加指標:
- `prediction_accuracy`: 予測 category と実測の一致率
- `predicted_outcome_by_candidate`: 候補ごとの予測分布

### 段階6: ToM + 内発駆動

追加指標:
- `internal_drive_avg`: 内発駆動スコアの平均
- `social_weight_effect`: social_weight が選択に寄与した回数
- **重要指標**: ToM 無視の選択が内発駆動高時に増えているか（MEMORY.md 思想遵守確認）

---

## 4. テスト手順

### 4.1 プロファイル作成

```bash
cd Noetic_seed/profiles
cp -r _template test_wm_phase1
# test_wm_phase1 は .gitignore で追跡外（profiles/* は _template 以外除外）
```

### 4.2 起動 (Windows bash)

```bash
cd Noetic_seed/profiles/test_wm_phase1
WM_DEBUG=1 .venv/Scripts/python.exe main.py
# または: export WM_DEBUG=1; python main.py
```

### 4.3 入力の与え方（ハイブリッド方式）

**パターン A: state.log に seed**
初期 state.json の log に device_input エントリを事前投入:
```json
{
  "id": "seed_ext1",
  "time": "2026-04-15 10:00:00",
  "tool": "[device_input]",
  "type": "external",
  "channel": "device",
  "result": "おはよう、見えてる？"
}
```
かつ pending にも:
```json
{
  "type": "external_message",
  "channel": "device",
  "id": "seed_ext1",
  "content": "おはよう、見えてる？",
  "timestamp": "2026-04-15 10:00:00",
  "priority": 3.0
}
```

**パターン B: WebSocket 手動介入**
WS クライアント (例: ブラウザ UI) 経由で途中メッセージを送る。タイミングは以下を推奨:
- サイクル 10 付近: 「ねえ、なにしてるの？」
- サイクル 30 付近: 「テストだよ、がんばって」

→ device_input の観測回数を十分稼ぐ。

### 4.4 50 サイクル走行

サイクル 50 到達 → Ctrl+C で停止 (`_force_exit_on_sigint` で即死)。

### 4.5 解析

```bash
cd Noetic_seed/profiles
python _template/design/wm_analyze.py test_wm_phase1
```

コンソールにサマリ、`test_wm_phase1/sandbox/wm_debug_summary.json` に詳細。

### 4.6 合格判定

M1 channel_match_ratio ≥ 0.80 なら段階1 合格。
不合格の場合は mismatch_cases を確認して原因調査（プロンプトの問題か、候補生成の問題か等）。

---

## 5. トラブルシューティング

### wm_debug.jsonl が生成されない
- `WM_DEBUG=1` が正しく渡っているか確認（`echo $WM_DEBUG`）
- `sandbox/` ディレクトリの書き込み権限

### M1 の分母が 0
- device pending になるイベントが発生していない
- → seed に device_input を仕込むか、WS で手動入力する

### M2 で device チャネル候補が出ない
- LLM① プロンプトに `[device]` タグがログ表示されているか確認
- `output_display` が allowed_tools に含まれているか（tool_level ≥ 0 で含まれる）

### mismatch が多すぎる（M1 < 0.5）
- LLM モデルが小さすぎる / プロンプトに device タグが見えていない
- `output_display` ツール説明の内容を確認
- 段階2 まで進めてエンティティモデルの助けを借りる必要あり

---

## 6. 後片付け

```bash
# テストプロファイル削除（必要なら）
rm -rf test_wm_phase1

# _template/sandbox/ に残った wm_debug.jsonl も（もしあれば）掃除
rm -f _template/sandbox/wm_debug.jsonl _template/sandbox/wm_debug_summary.json
```

---

## 7. 使用例: 段階1 テスト実行フロー

```bash
# 1. プロファイル作成
cd Noetic_seed/profiles
cp -r _template test_wm_phase1

# 2. seed 投入（state.json を編集）
# ... device_input の log + pending エントリを仕込む ...

# 3. 起動
cd test_wm_phase1
WM_DEBUG=1 ../../../.venv/Scripts/python.exe main.py

# 4. 約50サイクル待つ（~25-50分、1サイクル30秒前後）
#    途中で WS 経由で追加メッセージを送る

# 5. Ctrl+C 停止

# 6. 解析
cd ..
python _template/design/wm_analyze.py test_wm_phase1

# 7. 合否確認
#    M1 >= 0.80 なら段階1 合格 → 段階2 実装へ
#    M1 < 0.80 なら mismatch_cases を確認して原因究明
```

---

## 8. 更新履歴

- 2026-04-15 v1: 初版。段階1 用に作成。段階2-6 での拡張項目を予約。
