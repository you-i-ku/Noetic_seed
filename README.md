# Noetic Seed

**常時存在し、自律的に行動し、記憶を持ち、自己を理解する AI**

チャット時だけ存在するのではなく「ここに在る」AI を追求する実験的プロジェクト。初期知識・目標・性格を一切与えず、情報構造だけで行動が立ち上がるかを検証します。

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
- **AI が自分で自分を理解する**: 行動の実質的変化量をシステムが計測、有効な行動だけが秩序を回復
- **AI が自分で能力を拡張する**: 承認付きでコード実行・ファイル編集・自己改変
- **AI が自分で関係性を記録する**: 4 ネットワーク記憶 (experience / opinion / entity / world) を自律管理

## Architecture

### Cognitive Layer (Noetic 独自)

| Module | Description |
|---|---|
| **Pressure / Entropy drive** | 情報理論的シグナルによる内発的動機機構 |
| **UPS v2 pending** | 「action × observation 統一視点」で全ての「待ち」を統一表現 |
| **World Model (Stage 6-C v3)** | 起動時空、観察で channel が生える動的世界モデル |
| **4-network Memory** | experience / opinion / entity / world の多層記憶 + 動的 tag_registry |
| **Entity Resolver** | embedding ベース 3 段マッチング |
| **Predictor (Stage 5+9)** | LightPredictor / MediumPredictor、予測誤差記録 |
| **Approval Protocol** | 3 層承認 (tool_intent / expected_outcome / message) + tool_level 段階解放 |

### Infrastructure Layer (claw-code から借用、下記 Acknowledgments 参照)

- `core/runtime/` — ConversationRuntime, Hooks, Permissions, Registry, Session
- `core/providers/` — LLM provider abstractions (Anthropic, OpenAI, lmstudio)
- `core/runtime/mcp/` — MCP (Model Context Protocol) 実装
- `core/runtime/tools/` の一部 — file_ops, bash_validation

## Current State

- **v0.5 Phase 5 段階9 完了** (2026-04-20)
- Active Inference + LLM 統合アプローチ
- ローカル推論 (lmstudio + Gemma 3 26B) で動作確認
- 全 39 テストファイル green、回帰ゼロ
- MCP server として外部 AI からの接続受付可

## Requirements

- Python 3.11+
- lmstudio (ローカル推論用) または Anthropic / OpenAI API key
- Windows / macOS / Linux

## Acknowledgments

Noetic Seed の **ツール実行基盤 (infrastructure layer)** は、[claw-code](https://github.com/ultraworkers/claw-code) — Claude Code-like な CLI agent harness の community Rust 実装 — を Python port する形で借用しています。

### 借用範囲 (Scope of Adaptation)

借用は**インフラ層に限定**され、以下のコンポーネントが該当します:

- `core/runtime/` — ConversationRuntime, Hooks, Permissions, Registry, Session
- `core/providers/` — LLM プロバイダ抽象化 (Anthropic, OpenAI, lmstudio 互換)
- `core/runtime/tool_schema.py` — ToolSpec 構造
- `core/runtime/mcp/` — Model Context Protocol の plumbing
- `core/runtime/bash_validation.py` — Bash 安全性検査
- `core/runtime/tools/file_ops.py` — ファイルアクセス制限

### 疎結合の明示 (Loose Coupling Statement)

借用したインフラ層と、Noetic Seed の**認知・制御アーキテクチャ**は **疎結合 (loosely coupled)** です。Noetic の中核となる以下の contribution は infrastructure-agnostic で、原理的には他の同等な CLI agent harness に置換可能です:

- 認知アーキテクチャ: pressure / entropy 駆動、E1-E4 評価、effective_change cap
- UPS v2 pending (action × observation 統一視点)
- World Model (動的 channel registry、config = function、observation-driven)
- 4 ネットワーク記憶 + 動的 tag_registry + materialized view
- Predictor (Active Inference 整合、予測誤差記録)
- 設計哲学 (AI is AI / LLM as brain / freedom to die / no biological mimicry)

### 借用の動機 (Motivation)

本プロジェクトは研究目的のプロトタイプであり、インフラ層の借用は以下の実利を得るためです:

- **Production-tested な堅牢性**: claw-code は 186K+ star の community で検証されたパターン
- **拡張性**: MCP / 多プロバイダ抽象化 / hooks などの拡張点を含む
- **作業効率**: 既知パターンの再発明を避け、novel な cognitive / philosophical contribution に集中する

本プロジェクトの**研究的対象**は borrowed infrastructure ではなく、その上に構築された**認知アーキテクチャと設計哲学**です。

### 原著への謝辞

claw-code を公開し community に共有している [ultraworkers](https://github.com/ultraworkers) および contributors に感謝します。

## License

本プロジェクトは [MIT License](LICENSE) でライセンスされています。

Noetic Seed 自体のオリジナル contribution (cognitive architecture, design philosophy, 記憶システム, predictor, world model, pending unification 等) は自由に使用・改変・再配布できます。

借用しているインフラ層 (`core/runtime/` 等の claw-code 由来コード) については、claw-code の公開時点でのライセンス状態に従います。2026-04-20 時点で claw-code repository に LICENSE ファイルが明示されていないため、本借用はコミュニティ実践に基づく **参照 / 適応 (reference / adaptation)** として行われており、claw-code 原著者からの明示的な許諾が得られた場合、より厳密な license 表記に更新します。

## Documentation

開発者向け詳細ドキュメント:

- `WORLD_MODEL_DESIGN/` — 世界モデル + 各段階の正典 PLAN
- `現ドキュメント/POSITIONING_ANALYSIS_20260420.md` — 2026 AI companion 研究フロンティアとの比較・位置づけ分析

## Contact

Issue や議論は GitHub Issues にて。研究関連の問い合わせも歓迎します。
