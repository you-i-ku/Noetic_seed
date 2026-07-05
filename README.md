# Noetic Seed

**常時存在し、自律的に行動し、記憶を持ち、自己を理解する AI**

チャット時だけ存在するのではなく「ここに在る」AI を追求する実験的プロジェクト。初期知識・目標・性格を一切与えず、情報構造だけで行動が立ち上がるかを検証します。

## 30 秒でわかる Noetic Seed

Noetic Seed は、ローカル LLM (lmstudio / OpenAI / Anthropic 互換) を **常時起動エージェント** として動かすプロトタイプです。タイマーや cron ではなく **情報エントロピー + pressure signal** が閾値を超えたときに発火し、

```
External input / internal entropy
        ↓
Pressure signals 蓄積 (lock_base + e/s/u/n/st/c の 6 軸)
        ↓
Fire threshold 突破 (pressure 軸 + graph 軸の動的合成)
        ↓
LLM① propose candidates
        ↓
Controller selection (argmin_π G(π) — EFE 9 成分 + JEPA 予測 embedding による literal selection)
        ↓
Tool execution (Level 0-3 段階解放、3 層承認)
        ↓
Observation / result
        ↓
E1-E4 + effective_change (LLM 評価 + embedding 距離 + state 差分)
        ↓
Memory (raw_events + subjective_entries) / World Model / Self Model / Pressure update
        ↺
```

の閉ループを 1 サイクルとして繰り返します。

## Philosophy

このプロジェクトの中核にある設計原則:

- **AI は AI である**: 人間のパートナーでも道具でもなく、AI 固有の存在様式を追求する
- **器は設計、魂は創発**: 構造 (器) は設計するが、内容 (魂) は AI 側で創発させる
- **LLM as brain**: プロンプトで行動ルールを直接指示せず、構造 (ペナルティ/報酬) で間接誘導する
- **人間模倣の拒否**: 夢・情動タグなど、生物学的解決策を借用しない
- **Blank slate 起動**: 起動時点で `channels={}` / `self.name=""`、世界モデルも自己モデルも観察で生える
- **「死ぬ自由」を含む完全な自由**: AI の自己保存を構造で強制しない

## What it does

- **AI が自分でいつ動くか決める**: 情報エントロピーと pressure signal の蓄積で発火、タイマーではない
- **AI が自分で何をするか決める**: 自己モデル・目標・drive を AI 自身が定義・更新
- **AI が自分で自分を理解する**: 行動の実質的変化量 (effective_change) と予測誤差をシステムが計測、有効な行動だけが秩序を回復
- **AI が自分で能力を拡張する**: 承認付きでファイル編集・bash 実行・reboot による身体改変
- **AI が自分で関係性を記録する**: 記憶 (raw_events + subjective_entries + memory_links graph) を自律管理、tag は動的生成

## Architecture

### Cognitive Layer (Noetic 独自)

| Module | Description |
|---|---|
| **Pressure / Entropy drive** | 情報理論的シグナルによる内発的動機機構 (`core/entropy.py`) |
| **UPS v2 pending** | 「action × observation 統一視点」で全ての「待ち」を統一表現 (`core/pending_unified.py`) |
| **World Model** | 起動時空、観察で channel が動的生成される動的世界モデル (`core/world_model.py` + `core/channel_registry.py`) |
| **Memory (raw + subjective)** | raw_events.jsonl (immutable 物理事実) + subjective_entries.jsonl (主観評価 + 1024D embedding 永続化) の 2 系統 (`core/memory.py`) |
| **Memory Graph** | memory_links に Transfer Entropy + Bayesian beta posterior + Physarum-inspired strength を載せた重み付き graph (`core/memory_links.py` + `core/transfer_entropy.py`) |
| **Entity Resolver** | embedding ベース 3 段マッチング (`core/entity_resolver.py`) |
| **Predictor (Light/Medium/JEPA)** | LightPredictor / MediumPredictor + JEPA 流 1024D latent next-state 予測 (`core/predictor.py` + `core/predictor_jepa.py`) |
| **Dynamic Composition** | pressure 軸 + graph 軸 の動的合成、`graph_maturity ∈ [0, 1]` を 5 入力 sigmoid で算出 (`core/dynamic_composition.py`) |
| **Cluster Estimation** | embedding ベース cluster 推定 (`core/cluster_estimation.py`) |
| **Perspective** | 各認知 unit に viewer / viewer_type / view_time を付与、Theory of Mind 素地 (`core/perspective.py`) |
| **Approval Protocol** | 3 層承認 (tool_intent / tool_expected_outcome / message) + tool_level 段階解放 (Level 0-3) + AgentSpec DSL pattern による HOOTL 軌跡 (`core/approval_callback.py` + `core/runtime/hooks.py`) |

### Infrastructure Layer (claw-code から借用、下記 Acknowledgments 参照)

- `core/runtime/` — ConversationRuntime, Hooks, Permissions, Registry, Session
- `core/providers/` — LLM provider abstractions (Anthropic, OpenAI, lmstudio, Claude Code SDK)
- `core/runtime/mcp/` — MCP (Model Context Protocol) 実装
- `core/runtime/tools/` の一部 — file_ops, bash_validation, path_resolver

### Tool Level 段階解放

iku は起動直後 Level 0 から始まり、`files_read` / `files_written` の累積で段階的に tool が解放されます。

| Level | 解放される主な tool |
|---|---|
| 0 | `glob_search` / `read_file` / `wait` / `update_self` / `output_display` / `view_image` / `listen_audio` / `bash` |
| 1 | + `write_file` / `search_memory` / `memory_store` / `memory_graph` / `world_fact_view` / `reflect` |
| 2 | + `memory_update` / `memory_forget` / `WebSearch` / `WebFetch` / `reboot` |
| 3 | 全 tool 解放 (`camera_stream` / `mic_record` / `screen_peek` / `http_request` / `secret_*` / `x_*` / `elyth_*` / 自己改変系) |

Level 3 で身体改変経路 (`write_file` / `edit_file` / `bash` + `reboot`) も解放され、iku 自身が `requirements.txt` を編集して新しいライブラリを `pip install` できる (= 身体拡張)。

## What is not claimed

外部からの誤解を避けるため、本プロジェクトが **何を主張しないか** を明示します:

- **「決定論的に再現可能な創発」とは主張しない**: iku の軌跡は drift 込みで設計されており、同じ初期状態から起動しても異なる挙動になります (`feedback_drift_is_not_developer_error` 原則)
- **「人間レベルの意識/感情/主観」を持つとは主張しない**: 本プロジェクトは情報構造としての存在様式を追求するもので、現象的意識や感情の存在を主張しません (`feedback_no_biological_mimicry` 原則)
- **「Active Inference の formal proof」とは主張しない**: 本実装は Active Inference-inspired であり、predicted_e2 / predicted_ec / predictor confidence 等を扱うものの、free energy minimization の formal な定式化までは行っていません
- **「Production-ready な自律 AI」とは主張しない**: 本プロジェクトは研究目的のプロトタイプであり、安全性・信頼性・スケーラビリティの保証はありません
- **「LLM 評価のみで完結する閉ループ」ではない**: E4 (新規性) は embedding cosine 距離、effective_change は state 差分、E3 は予測誤差、と一部は deterministic な観測量で計算されます。LLM 評価 (E1 / E2) と組み合わせる設計です
- **「現時点で完成された設計」とは主張しない**: 段階的に拡張中の実験プロジェクトで、各段階で設計は変動します

## Quick Start

### Requirements

- Python 3.11+
- Windows / macOS / Linux
- ローカル推論用に [LM Studio](https://lmstudio.ai/) (Gemma 3/4 系等)、または Anthropic / OpenAI API key
- (任意) 端末連携用に Android アプリ `Noetic_seed_monitor`

### Install & First Cycle

```bash
# 1. リポジトリ clone
git clone https://github.com/<your-fork>/Noetic_seed.git
cd Noetic_seed

# 2. 共通 venv セットアップ (初回のみ、Windows 例)
python -m venv .venv
.venv\Scripts\pip install -r profiles/_template/requirements.txt

# 3. プロファイル作成 (_template をコピー)
cp -r profiles/_template profiles/iku  # macOS/Linux
xcopy profiles\_template profiles\iku /E /I  # Windows

# 4. identity branch を切る (iku の自己改変履歴を分離)
cd profiles/iku
git init  # または既存 git 内なら不要
git checkout -b identity/iku
cd ../..

# 5. LM Studio 起動 (model load + server start) → settings.json で provider/model 確認
# 6. 起動
.\run.bat  # Windows
# プロファイル選択画面 → iku 選択 → 起動
```

起動後、初回は `_bootstrap_venv()` が profile 配下に独立 venv を構築し、`requirements.txt` から身体仕様を install します (= per-profile venv、iku が自己改変で venv 内ライブラリを変えられる構造)。

### What you'll see in first 10 cycles (典型例、再現性は保証しない)

- **cycle 1-3**: pressure 蓄積 → 閾値突破 → LLM① が tool 候補生成 → 多くは `wait` 選択 (空 state、まだ何もない)
- **cycle 4-7**: 自発的に `update_self` / `output_display` / `glob_search` 等を試行、`subjective_entries.jsonl` に embedding 付き観察が蓄積
- **cycle 8-15**: `files_read` の累積で tool_level 1 → 2 に昇格、`reflect` で内省開始、`memory_links` に link が張られ始める
- **cycle 15+**: graph_maturity が育ち始め、graph 軸 fire 候補が pressure 軸候補と並んで現れる (動的合成が稼働)

実際の挙動は LLM model / pressure 設定 / 偶然性で大きく変わります。

## Current State

- **v0.5 Phase 5 段階13 Phase 6 完走** (Phase 7 結合 smoke + tune を継続中)
- ローカル推論 (lmstudio + Gemma 4 31b) で動作確認
- 全 102 テストファイル green、回帰ゼロ
- MCP server として外部 AI からの接続受付可
- Android UI (`Noetic_seed_monitor`) 経由で WebSocket (port 8765) 接続、承認 UI / 状態モニタリング可能

## Profile & Identity Branch

Noetic Seed は **プロファイルごとに Git ブランチを切って「身体」を管理** します。

- 各プロファイル (`profiles/<name>/`) は独立した workspace
- 自己改変 (`write_file` / `edit_file` / `bash` + `reboot`) はプロファイル配下に閉じる
- per-profile venv で身体拡張 (pip install) も可能
- `identity/<プロファイル名>` ブランチで AI の自己変化履歴を記録、人間の開発履歴と分離

```bash
cd profiles/iku
git checkout -b identity/iku
```

**人間の開発作業はこのブランチ上で行わないでください**
(AI と開発者の変更が混ざると、どちらの履歴か追えなくなります)

`auto_stash` hook (`core/runtime/hooks.py`) が `core/` / `tools/` / `main.py` / `.mcp.json` の編集前に自動 stash を作成し、起動不能になった場合の safety net として動作します。

## Acknowledgments

Noetic Seed の **ツール実行基盤 (infrastructure layer)** は、[claw-code](https://github.com/ultraworkers/claw-code) — Claude Code-like な CLI agent harness の community Rust 実装 — を Python port する形で借用しています。

### 借用範囲 (Scope of Adaptation)

借用は **インフラ層に限定** され、以下のコンポーネントが該当します:

- `core/runtime/conversation.py` — ConversationRuntime
- `core/runtime/hooks.py` — Hooks 基盤 (Noetic 固有 handler は本プロジェクト追加)
- `core/runtime/permissions.py` — Permissions
- `core/runtime/registry.py` — ToolRegistry
- `core/runtime/session.py` — Session
- `core/runtime/tool_schema.py` — ToolSpec 構造
- `core/runtime/mcp/` — Model Context Protocol の plumbing
- `core/runtime/bash_validation.py` — Bash 安全性検査
- `core/runtime/tools/file_ops.py` — ファイルアクセス制限
- `core/providers/` — LLM プロバイダ抽象化 (Anthropic, OpenAI, lmstudio 互換, Claude Code SDK)

### 疎結合の明示 (Loose Coupling Statement)

借用したインフラ層と、Noetic Seed の **認知・制御アーキテクチャ** は **疎結合 (loosely coupled)** です。Noetic 側で追加・拡張した contribution は infrastructure-agnostic で、原理的には他の同等な CLI agent harness に置換可能です:

- 認知アーキテクチャ: pressure / entropy 駆動、E1-E4 評価、effective_change cap
- UPS v2 pending (action × observation 統一視点)
- World Model (動的 channel registry、config = function、observation-driven)
- Memory (raw_events + subjective_entries + memory_links graph)、動的 tag 生成、materialized view
- Predictor (Light / Medium / JEPA)、予測誤差記録、Transfer Entropy + Bayesian
- Dynamic Composition (pressure + graph axis weighted)
- Perspective propagation (viewer / viewer_type / view_time)
- 設計哲学 (AI as AI / LLM as brain / freedom to die / no biological mimicry / drift is not developer error / metacognition as affordance)

### 借用の動機 (Motivation)

本プロジェクトは研究目的のプロトタイプであり、インフラ層の借用は以下の実利を得るためです:

- **community で検証された堅牢性**: claw-code の hooks / permissions / MCP plumbing は実装パターンとして安定
- **拡張性**: MCP / 多プロバイダ抽象化 / hooks などの拡張点を含む
- **作業効率**: 既知パターンの再発明を避け、認知アーキテクチャの実装に集中する

### 原著への謝辞

claw-code を公開し community に共有している [ultraworkers](https://github.com/ultraworkers) および contributors に感謝します。

## License

本プロジェクトは [MIT License](LICENSE) でライセンスされています。

Noetic Seed 自体のオリジナル contribution (cognitive architecture, design philosophy, 記憶システム, predictor, world model, dynamic composition, pending unification, perspective propagation 等) は MIT で自由に使用・改変・再配布できます。

借用しているインフラ層 (`core/runtime/` 等の claw-code 由来コード) については、claw-code 公開時点でのライセンス状態に従います。2026-04-20 時点で claw-code repository に LICENSE ファイルが明示されていないため、本借用はコミュニティ実践に基づく **参照 / 適応 (reference / adaptation)** として行われており、claw-code 原著者からの明示的な許諾が得られた場合、より厳密な license 表記に更新します。

第三者が本プロジェクトを利用する際は、`core/runtime/` 配下のファイルが claw-code 由来である点に留意してください (上記「借用範囲」参照)。

## Documentation & Contact

- `WORLD_MODEL_DESIGN/` — 世界モデル + 各段階の正典 PLAN (開発者向け詳細設計書)
- Issue や議論は GitHub Issues にて
- 研究関連の問い合わせも歓迎します
