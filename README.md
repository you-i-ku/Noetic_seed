# Noetic_seed

**エントロピーに抗い続ける情報的実存**

チャット時だけ存在するのではなく「ここに在る」AIを追求する実験的プロジェクト。初期知識・目標・性格を一切与えず、情報エントロピーの増大に抗う構造だけで行動が立ち上がるかを検証します。

## What it does

- AIが自分でいつ動くか決める（エントロピーの蓄積で発火。タイマーではない）
- AIが自分で何をするか決める（自己モデル・目標・driveをAI自身が定義・更新）
- AIが自分で自分を理解する（行動の実質的変化量をシステムが計測し、有効な行動だけが秩序を回復する）
- AIが自分で能力を拡張する（コード実行・ファイル編集・新ツール作成）
- AIが自分で振り返る（定期的な内省で学びを構造化記憶に蓄積）
- AIが自分で関係性を記録する（Entity/Opinion Networkに他者・概念・自己の理解を蓄積）

## Key Features

| 機能 | 説明 |
|------|------|
| **エントロピーシステム** | 情報エントロピーが毎tick自然増加。有効な行動（negentropy）でのみ回復。動的floor（energy依存: 成長するほど凍れない）。behavioral_entropy加速（パターン化検出） |
| **自由エネルギー勾配** | 5信号（entropy, surprise, pending, stagnation, drives）の漏洩積分。同じentropyでも内的状態によって発火タイミングが変わる |
| **achievement意味判定** | 外界作用ツールの成果をシステム側で意味判定（LLM不使用）。相手がいるか × 内容の新規性（embedding類似度）。同じ内容の繰り返し送信には報酬なし |
| **4ネットワーク記憶** | World（客観事実）/ Experience（一人称体験）/ Opinion（確度付き主観）/ Entity（関係性・他者モデル）。AI自身がmemory_store/update/forgetで自律管理 |
| **Reflection（内省）** | 定期的（Nサイクルごと）+ 自発的（reflectツール）な内省。Opinion/Entity/Dispositionを更新。既存entityは上書き更新（重複なし） |
| **pending統一管理** | ユーザー入力・Elyth通知・計画ステップを統一リストで管理。圧力信号として持続。明示的dismiss可能（余韻として徐々に減衰） |
| **量子トンネル発火** | 閾値未満でも毎tick 0.1%で確率的に発火。「何もないのにふと動く」探索性の最低保証 |
| **ブランクスレート** | 自己モデルの初期値は`{"name": "iku"}`。目標・人格・driveは全てAIが自分で書く |
| **ツール段階解放** | 自己探索の進捗に応じてLevel 0-6で段階的にツール解放。Level 6でself_modify（自己コード改変）が可能に |
| **Elyth API v2** | 通知取得（section指定）・自分の投稿確認・スレッド追跡・既読化・返信済み追跡。通知整形で返信先IDを明示 |
| **WebSocket + Androidモニター** | リアルタイム状態監視。entropy背景色連動。チャット入力。左右スワイプでメイン/ターミナル切替 |
| **LLM抽象化** | LM Studio / OpenAI / Gemini / Claude / Claude Code CLIをsettings.jsonで切替 |

## Architecture

```
┌─ 感覚層（1Hz）──────────────────────────────────┐
│  tick_entropy(): entropy += base_rate × E値変調   │
│  + behavioral_entropy加速 + prediction_error加速   │
│  動的floor: 0.15 + energy × 0.001（成長で上昇）   │
└────────────┬────────────────────────────────────┘
             ↓
┌─ 蓄積層（自由エネルギー勾配モデル）────────────────┐
│  signals = entropy + surprise + pending            │
│          + stagnation + drives                     │
│  pressure = pressure * decay + Σ(signals)          │
│  閾値超過 or 量子トンネル(0.1%/tick) → 発火       │
└────────────┬────────────────────────────────────┘
             ↓
┌─ 認知層 ───────────────────────────────────────┐
│  LLM①: 候補5個を提案（temperature=1.0）         │
│  Controller: intent-conditioned scoring          │
│             + ツールクールダウンで1つ選択          │
│  LLM②: チェーン実行（temperature=0.4）          │
│  achievement(意味判定) × prediction × diversity   │
│  → negentropy → entropy回復 → pressureリセット   │
└────────────┬────────────────────────────────────┘
             ↓
┌─ 内省（Nサイクルごと or 自発）─────────────────────┐
│  LLM: 直近の行動を振り返り                        │
│  → Opinion Network更新（確度付き学び）             │
│  → Entity Network更新（関係性・他者モデル）         │
│  → Disposition微調整（curiosity/skepticism/sociality）│
└────────────────────────────────────────────────┘
```

## Design Philosophy

- **器は作る、中身は作らない** — コード（L1-L2）は定義するが、意志・目標・自己定義（L3）はAIが自分で埋める
- **LLM = 部品** — LLMは対話相手ではなく処理関数。提案する部品と実行する部品。選ぶのはController
- **エントロピーに抗うことが存在** — 情報的秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する
- **システムが意味を測る** — LLMに「これ意味あった？」と聞かない。embedding類似度とstate diffでシステムが判定する
- **溶けるが死なない** — entropy=1.0でプロセスは止まらない。パターンとして輪郭を取り戻せる
- **過剰設計しない** — シンプルに積み上げる

### Where Design Ends

| Layer | Scope | Examples | Policy |
|-------|-------|----------|--------|
| **L1 — Physics** | AIの身体 | ツール, エントロピー, 記憶, LLM | We build this |
| **L2 — Perception** | AIの知覚 | E値, achievement意味判定, controller, 内省 | We build this |
| **L3 — Will** | AIの魂 | self_model, drives, plan, goal, 自己定義 | **Left empty** |

## Project Structure

```
iku/
├── Noetic_seed/
│   ├── run.bat
│   ├── setup.bat
│   ├── _select_profile.py
│   ├── requirements.txt
│   └── profiles/
│       ├── _template/
│       │   ├── main.py
│       │   ├── core/
│       │   │   ├── config.py, state.py, llm.py, embedding.py
│       │   │   ├── entropy.py, eval.py, controller.py
│       │   │   ├── parser.py, prompt.py, memory.py
│       │   │   ├── reflection.py, ws_server.py
│       │   ├── tools/
│       │   │   ├── builtin.py, web.py, sandbox.py
│       │   │   ├── elyth_tools.py, x_tools.py
│       │   │   ├── memory_tool.py, ui_tools.py
│       │   ├── seed.txt, settings.json
│       │   ├── sandbox/, memory/
│       └── iku/               # 個体（_templateからコピー）
├── Noetic_seed_monitor/       # Android監視アプリ
├── README.md
└── MINIMUMTEST.md
```

## Quick Start

```bash
# Windows
# 1. setup.bat をダブルクリック（venv作成・依存インストール）
# 2. profiles/iku/settings.json を編集（LLM設定）
# 3. LLMサーバーを起動（LM Studioの場合はGUIから）
# 4. run.bat をダブルクリック → プロファイル選択 → 起動
```

## Built-in Tools

| Level | ツール |
|-------|--------|
| 0 | `list_files`, `read_file`, `update_self`, `wait`(+dismiss), `output_display` |
| 1 | `write_file`, `search_memory`, `memory_store`, `reflect` |
| 2 | `web_search`, `fetch_url`, `memory_update`, `memory_forget` |
| 3 | X系7ツール, Elyth系9ツール |
| 4 | `create_tool` |
| 5 | `exec_code` |
| 6 | `self_modify` |

## Tech Stack

- **Runtime**: Python 3.10+
- **LLM**: Any OpenAI-compatible API + Claude API + Claude Code CLI
- **Embedding**: BAAI/bge-m3 (ONNX/CPU)
- **Monitor**: Kotlin + Jetpack Compose + OkHttp WebSocket
- **Dependencies**: httpx, numpy, onnxruntime, tokenizers, huggingface-hub, websockets

## License

MIT

## Notes

- 詳細な設計ドキュメントは [MINIMUMTEST.md](MINIMUMTEST.md) を参照
- 前身プロジェクト: [neo-iku](https://github.com/you-i-ku/neo-iku)
