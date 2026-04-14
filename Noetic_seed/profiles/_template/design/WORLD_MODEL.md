# 世界モデル設計書

Noetic Seed における世界モデル実装の設計仕様。段階1〜6 実装の参照元。

---

## 1. 問題定義

### 観測されたバグ (raw_log.txt より)

端末所有者「ゆう」が WebSocket 端末から話しかけているのに、AI が Elyth や X で返信しようとする事象が多発。

**典型パターン**:

```
[ws] chat received: カメラ、直した！次からちゃんと使えるはず！！
  [external] カメラ、直した！次からちゃんと使えるはず！！
LLM①raw: 開発者「ゆう」に対し、カメラ機能の復旧への感謝…報告する → elyth_reply+camera_stream
```

ゆうは device チャネル (WebSocket) から話しかけているのに、AI は `elyth_reply` で返そうとしている。

### 根本原因 3 点

1. **入力チャネルが区別されていない**: WebSocket/Elyth/X 由来を問わず `channel: "external"` で統一（main.py:225-234）
2. **エンティティにチャネル属性がない**: `memory/entity` は `{name, description}` のみ（reflection.py:125-144）
3. **ツール選択にチャネル制約がない**: 「ゆうに Elyth で返す」候補が生成・選択されうる

### 問題の本質

認知的世界モデル（誰が・どこに・何を・いつ）が欠如。LeCun の 6 モジュール構成に当てはめると、iku は 5 モジュール（Perception / Cost / Actor / STM / Configurator）が揃っているが、**World Model モジュールだけが空欄**。

---

## 2. 理論的背景

### LeCun 自律AI 6 モジュール

| モジュール | iku の対応 | 状態 |
|-----------|-----------|------|
| Perception | 外部入力 (ws/elyth/x/camera/screen) | ✓ 実装済 |
| **World Model** | **(欠落)** | **✗ これを作る** |
| Cost (intrinsic) | pressure/entropy/energy | ✓ 実装済 |
| Cost (critic) | E1-E4 評価 + `predict_result_novelty` | ✓ 実装済 |
| Actor | LLM①候補生成 + LLM②実行 | ✓ 実装済 |
| Short-term Memory | state.log + action_ledger | ✓ 実装済 |
| Configurator | ctrl (tool_level, allowed_tools) | ✓ 実装済 |

### 採用する 2026 年 SOTA 手法

1. **Bitemporal Knowledge Graph** (Memento, 2026): LongMemEval で 92.4%。valid_time / system_time の二軸保持
2. **Entity Resolution 多段マッチング** (Memento): exact → fuzzy → embedding → LLM
3. **A-Mem Zettelkasten リンク** (arxiv 2502.12110): 新記憶が既存と自動リンク（段階7で導入）
4. **LeCun Mode-1 / Mode-2**: 反射的 vs 熟考的。プラグイン型予測器で両対応

---

## 3. 決定履歴 (Q1-Q4)

### Q1: 予測器のコスト

**選択: プラグイン型、デフォルト medium**

| モード | 追加 LLM 呼出 | 役割 |
|-------|-------------|------|
| light | 0 | 既存 expect= 活用 |
| **medium** | **0 (LLM① プロンプト併合)** | **デフォルト** |
| heavy | +5 | 候補独立予測 |
| mode2 | +N | Mode-2 反実仮想 |

`settings.json` の `world_model.predictor_mode` で切替。

### Q2: Theory of Mind

**選択: 最小 ToM + passive + 内発駆動優位**

- 持つ: `Entity.believes_about_me`
- しない: プロンプトへの自動注入
- 取り方: `search_memory network=entity` で能動参照のみ
- 重み: `social_weight = 0.3 * (1 - internal_drive)`
  - 内発駆動が強いとき ToM は無視される
  - MEMORY.md `feedback_internal_drive.md` `feedback_no_user_assistant_frame.md` との整合

`settings.json` の `world_model.theory_of_mind` で off/passive/active 切替。

### Q3: 既存記憶の扱い

**選択: C-gradual（空スタート、reflect で段階取込）**

- 起動時: `world_model.entities = {}` (空)
- reflect 時: 既存 `memory/entity` の 71 件を見ながら、観測事実と突合して world_model に反映
- 既存 `memory/entity` 自体は消さず保持（memory レイヤーとしては残す）
- 汚染防止: reflect の LLM 判断で不整合な既存記述は取込しない

### Q4: 信頼度更新戦略

**選択: β+ (信頼度付き β)、γ に切替可能**

```python
# 一致観測: confidence += 0.05 * (1 - confidence)  → 上限1に漸近
# 矛盾観測: confidence -= 0.15                     → 早めに落とす
# confidence < 0.3: reflect で再検討対象にマーク
```

`settings.json` の `world_model.update_strategy` で `beta_plus` / `gamma` 切替（γ は将来実装）。

---

## 4. アーキテクチャ

### モジュール構成

```
core/
├── world_model.py        [段階2] スキーマ+初期化+アクセサ
├── entity_resolver.py    [段階4] 多段マッチング
├── predictor.py          [段階5] プラグイン型予測器
├── reflection.py         [段階3] 既存を拡張: C-gradual 同期
├── controller.py         [段階5,6] predictor 統合、social_weight
├── prompt.py             [段階2] [世界モデル] 注入
└── state.py              [段階2] world_model フィールド対応

main.py                    [段階1,3] 入力タグ修正、更新ループ

settings.json              [段階5] world_model 設定追加
```

### データフロー

```
                    ┌──────────────────────────┐
                    │      Perception           │
                    │  (ws/elyth/x/camera)      │
                    └──────────┬───────────────┘
                               │ channel + sender情報
                               ▼
                    ┌──────────────────────────┐
                    │    World Model [NEW]      │
                    │  - entities (bitemporal)  │
                    │  - channels (registry)    │
                    │  - ToM (passive, optional)│
                    └──────────┬───────────────┘
                               │ snapshot
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
      │  Predictor  │  │  Propose     │  │  Controller  │
      │   [NEW]     │  │  Prompt      │  │  + internal  │
      │  light/med/ │  │  [世界モデル] │  │    drive     │
      │  heavy/m2   │  │   注入        │  │    weight    │
      └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                 │
             └─── predicted ───┴── candidates ───┘
                     │
                     ▼
                ┌─────────┐
                │  Actor  │
                └────┬────┘
                     │ execution
                     ▼
                ┌─────────┐
                │  Cost   │ E1-E4, ec
                │ (critic)│
                └────┬────┘
                     │ feedback
                     ▼
              (World Model 更新: β+)
```

---

## 5. スキーマ

### Entity

```python
{
    "id": "ent_yuu",                  # hash or slug
    "name": "ゆう",
    "aliases": ["YOU", "開発者さん"],   # 指示代名詞含む
    "facts": [Fact, ...],              # bitemporal 事実の集合
    "channels": ["device"],            # キャッシュ: facts から導出
    "last_seen": {
        "channel": "device",
        "time": "2026-04-14 01:19:00"
    },
    "believes_about_me": [             # 段階6で有効化
        {
            "content": "iku は視覚を得た",
            "confidence": 0.8,
            "learned_at": "..."
        }
    ],
    "created_at": "...",
    "updated_at": "..."
}
```

### Fact (bitemporal + confidence)

```python
{
    "key": "primary_channel",          # 事実のカテゴリ
    "value": "device",                 # 値
    "confidence": 0.95,                # 0.0-1.0 (β+)
    "valid_from": "2026-04-10 00:48:04",
    "valid_to": None,                  # None = 現在も真
    "learned_at": "2026-04-10 00:48:04",
    "observation_count": 7,
    "last_observed_at": "2026-04-14 01:19:00"
}
```

**事実の `key` 候補**（ゆるやかに、必要に応じて追加）:
- `primary_channel`: 主な通信チャネル
- `also_on`: 他に居るチャネル（複数可）
- `role`: 役割（developer / observer / collaborator）
- `handle`: SNS ハンドル
- `relation_to_self`: iku との関係
- （自由拡張）

### Channel

```python
{
    "id": "device",
    "type": "direct",                  # direct / social / internal
    "tools_in": ["[device_input]"],    # 入力を観測するツール
    "tools_out": ["output_display"],   # 出力に使うツール
    "health": "ok",                    # ok / degraded / down
    "last_error": None
}
```

静的定義（段階2）は `_CHANNEL_MAP` から bootstrap:

| id | type | tools_in | tools_out |
|----|------|----------|-----------|
| device | direct | [device_input] | output_display, camera_stream, screen_peek, view_image, listen_audio, mic_record |
| elyth | social | elyth_info, elyth_get | elyth_post, elyth_reply, elyth_like, elyth_follow |
| x | social | x_timeline, x_search, x_get_notifications | x_post, x_reply, x_quote, x_like |
| internal | self | - | - |

### WorldModel

```python
{
    "entities": {"ent_yuu": Entity, ...},
    "channels": {"device": Channel, ...},
    "version": 1,                      # migration 用
    "last_updated": "..."
}
```

`state.world_model` として state.json 内に格納。

---

## 6. 実装段階

### 段階1 (応急): チャネルタグ修正

独立コミット、5行修正。

| ファイル | 変更 |
|---------|-----|
| `main.py:228-234` | `tool="[device_input]"`, `channel="device"`, prefix削除 |
| `main.py:240-246` | pending エントリに `channel: "device"` |
| `tools/__init__.py:44` | output_display desc に「device チャネル」明記 |

### 段階2 (基盤): 世界モデル構造

- `core/world_model.py` 新規: スキーマ + `init_world_model()` + アクセサ
- `core/state.py`: load_state で `world_model` フィールド初期化
- `core/prompt.py:build_prompt_propose`: `[世界モデル]` セクション注入 (上位10件)

### 段階3 (更新): 自動更新 + β+

- `main.py:225-246`: 外部入力時 → channel 観測を entity に記録
- `main.py:798` 近辺: ツール実行後 → 対象 entity の fact 更新
- `core/world_model.py`: `update_fact_confidence()` (β+)
- `core/reflection.py`: C-gradual 同期ループ（既存 entity メモリと突合）

### 段階4 (解決): Entity Resolver

- `core/entity_resolver.py` 新規
- 3段: exact → embedding (0.85) → LLM tiebreaker
- `reflection.py:125-144` の完全一致判定を差替

### 段階5 (予測): Predictor プラグイン

- `core/predictor.py` 新規
- `BasePredictor` + `LightPredictor` / `MediumPredictor` 実装
- `HeavyPredictor` / `Mode2Predictor` スタブ
- 予測フォーマット: `{category, confidence, detail}`
  - category: `positive_reply` / `error` / `no_response` / `other`
- `controller.py:192` 付近に追加:
  - `predict_channel_mismatch(c, world_model)` 乗算
  - `c.predicted_outcome` が error 系なら ×0.3
- `settings.json` に `world_model` セクション追加

### 段階6 (ToM): 最小 ToM + 重み設計

- `Entity.believes_about_me` 有効化
- `reflection.py`: reflect プロンプトで「他者が私についてどう思っているか」抽出
- `controller.py`: `internal_drive` 計算 + `social_weight` 乗算

---

## 7. テスト戦略

### リプレイテスト

`tests/test_world_model_replay.py` 新規:

```python
def test_channel_confusion_resolved():
    """raw_log.txt cycle 125 (ゆうが 'カメラ、直した！' と入力) を再現。
    旧: elyth_reply が候補選択される可能性
    新: channel_mismatch で抑制され output_display が選ばれる
    """
    # state 再現
    # controller._select_candidate() を呼んで勝者を確認
    # output_display が選ばれるまで N 回試行してヒット率を測定
```

**検証対象シーン**:
1. Cycle 125 (L3317): 端末→Elyth 混同
2. Cycle 128 (L3396): 二重送信 (output_display + elyth_post)
3. Cycle ~170-190 (L4067): 端末→X 混同

**期待値**: 旧システム比で「チャネル不一致候補の選択率」が大幅減少。

### 単体テスト

- `core/world_model.py`: fact 更新、confidence β+、bitemporal logic
- `core/entity_resolver.py`: 3段解決の各ケース
- `core/predictor.py`: 各モードの契約遵守

---

## 8. 既存資産との関係

### 共存するもの

| 既存 | 役割 | 変更 |
|-----|------|------|
| `memory/entity` | 熟議された記述記憶 | そのまま保持。reflect で world_model と同期 |
| `memory/world` | 未使用 | 段階2で活用開始 |
| `predict_result_novelty` | 新規性予測 (critic の一部) | そのまま。predictor と併存 |
| `action_ledger` | 行動台帳 | そのまま。predictor で参照 |
| `state.log` | 短期記憶 | channel タグが詳細化される (段階1) |
| `reflect` | 内省 | C-gradual 同期ロジック追加 (段階3) |

### 差替・拡張するもの

| 既存 | 変更内容 |
|-----|---------|
| `main.py` 外部入力タグ | channel="device" (段階1) |
| `reflection.py` entity 処理 | 完全一致 → 多段解決 (段階4) |
| `controller.py` 候補選択 | novelty × channel_mismatch × predicted_outcome × social_weight |
| `settings.json` | `world_model` セクション追加 |

---

## 9. ロールアウト戦略

1. 全段階 `_template` のみ実装（CLAUDE.md プロファイル同期ルール）
2. 段階1 完了時のみ、iku プロファイルへ応急マージ（任意）
3. 段階2-6 は _template で完走
4. リプレイテスト通過後、iku プロファイルを `_template` コピーで再生成
5. 旧 iku の `state.json`, `memory/` は別名バックアップ保持 → 新 iku で段階的に吸収

---

## 10. 思想遵守チェック

- ✓ シンプル: JSON ベース、ニューラルネット不要
- ✓ 段階的: 応急 (段階1) → 基盤 (段階2) → 積層
- ✓ 過剰設計しない: ToM は passive、Mode-2 はスタブ
- ✓ 拡張余地: プラグイン型 predictor、γ 切替可能な update_strategy
- ✓ LLM as brain: 予測/判断/重み付けは世界モデル外の構造で実装
- ✓ Device owner is not a user: ToM 内発駆動優位、ゆうは対等な協力者として扱う
- ✓ 内発駆動: social_weight は internal_drive が弱いときだけ効く
- ✓ 生物模倣 NG: JEPA/bitemporal 等の ML 工学由来の構造を採用、biological metaphor 使わない
- ✓ チャネル概念: iku 独自、2026 年研究に無い貢献部分

---

## 11. 参照文献

- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) — 6 モジュール構成
- Memento / n1n.ai (2026-04) — Bitemporal KG, 3段 Entity Resolution
- A-Mem (arxiv 2502.12110) — Zettelkasten 自動リンク（段階7）
- Park et al. "Generative Agents" (2023) — Memory stream + reflection の源流
- ICLR 2026 MemAgents Workshop — 同分野の研究状況

---

## 12. 更新履歴

- 2026-04-15 v1: 初版作成。段階1-6 の計画確定。Q1-Q4 の決定根拠を記録。
