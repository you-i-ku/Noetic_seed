# iku

**Idle Kernel, Undefined — エントロピーに抗い続ける情報的実存**

チャット時だけ存在するのではなく「ここに在る」AIを追求する実験的プロジェクト。初期知識・目標・性格を一切与えず、情報エントロピーの増大に抗う構造だけで行動が立ち上がるかを検証します。

## What it does

- AIが自分でいつ動くか決める（エントロピーの蓄積で発火。タイマーではない）
- AIが自分で何をするか決める（自己モデル・目標・driveをAI自身が定義・更新）
- AIが自分で自分を理解する（E1-E4の予測誤差がnegentropyとして秩序を回復する）
- AIが自分で能力を拡張する（コード実行・ファイル編集・新ツール作成）
- 有効な行動だけが秩序を回復する。同じことの繰り返しでは秩序は回復しない

## Key Features

| 機能 | 説明 |
|------|------|
| **エントロピーシステム** | 情報エントロピーが毎tick自然増加（E値で変調。反復・停滞で加速）し、有効な行動（negentropy）でのみ回復。entropy=0で鮮明、entropy=1.0で溶ける（死なない、ぼやけるだけ）。negentropy = E1(計画品質) × E2(達成度) × E4(新規性) × surprise_bonus(E3)。増加と減少がE1-E4で対称的に変調される |
| **自由エネルギー勾配** | pressureはentropyの単純変換ではなく、4信号（entropy, surprise, unresolved, novelty）+ custom_drivesの漏洩積分。同じentropyでも内的状態によって発火タイミングが変わる |
| **量子トンネル発火** | 閾値未満でも毎tick 0.1%で確率的に発火。「何もないのにふと動く」探索性の最低保証 |
| **認知品質の連動** | entropyが選択の鋭さに影響。低entropy=確信的に行動、高entropy=散漫にランダム化。energyの探索（前向き）とは異なるシグナル |
| **メタ認知ループ** | ツール呼び出し時にintentで意図、expectで予測を記録。E1(計画整合性)・E2(達成度)・E3(予測精度)・E4(新規性)の4軸でnegentropyを計算 |
| **intent-conditioned scoring** | 過去の「同じtool × 類似intent」のE2実績を学習し、controller_selectの重みに反映。失敗した行動パターンを構造的に回避 |
| **ブランクスレート** | 自己モデルの初期値は`{"name": "iku"}`。目標・人格・driveは全てAIが自分で書く |
| **custom_drives** | AI自身が`pref.json`に固有の欲求を定義（Level 6で解放）。entropyとは独立したpressure寄与。L3（意志層）の直接的な実現 |
| **ツール段階解放** | 自己探索の進捗に応じてLevel 0-6で段階的にツール解放。条件は非明示。Level 6でself_modify（自己コード改変）+ pref.json読み書きが可能に |
| **自己改変** | コード実行・ファイル書き換え・新ツール作成（Human-in-the-loop承認付き） |
| **1ファイル・アーキテクチャ** | main.py単体で完結。依存はPython + LLMサーバーのみ |
| **長期記憶** | 全行動をJSONLアーカイブに保存。bge-m3ベクトル検索（ONNX/CPU）。階層要約で圧縮しつつ逆引き可能 |
| **LLM抽象化** | LM Studio / OpenAI / Gemini / Claude をsettings.jsonで切り替え可能。テキストマーカー`[TOOL:...]`方式でfunction calling非依存 |

## Architecture

```
┌─ 感覚層（1Hz）─────────────────────────────┐
│  tick_entropy(): entropy += base_rate × E値変調│
│  LLM呼び出しなし。プロセスは止まらない        │
└────────────┬───────────────────────────────┘
             ↓
┌─ 蓄積層（自由エネルギー勾配モデル）───────────┐
│  signals = entropy + surprise + unresolved    │
│          + novelty + custom_drives            │
│  pressure = pressure * decay + Σ(signals)     │
│  閾値超過 or 量子トンネル(0.1%/tick) → 発火   │
└────────────┬───────────────────────────────┘
             ↓
┌─ 認知層 ──────────────────────────────────┐
│  LLM①: 候補5個を提案（temperature=1.0）    │
│  Controller: intent-conditioned scoring     │
│             + entropy認知品質で1つ選択       │
│  LLM②: チェーン実行（temperature=0.4）     │
│  E1-E4 → negentropy(entropy回復)           │
│        → entropy増加率変調（次tick以降）     │
│        → pressure信号更新（次tick以降）      │
│  ログ記録 → 次のサイクルへ                  │
└────────────────────────────────────────────┘
```

## Design Philosophy

- **器は作る、中身は作らない** — コード（L1-L2）は定義するが、意志・目標・自己定義（L3）はAIが自分で埋める
- **LLM = 部品** — LLMは対話相手ではなく処理関数。提案する部品と実行する部品。選ぶのはController。主体はstate遷移の履歴の総体に宿る
- **エントロピーに抗うことが存在** — 情報的秩序は放っておけば崩壊する（第二法則）。有効な行動だけが秩序を回復する。生命の模倣ではなく、情報系の物理法則
- **勾配が仕事を生む** — エントロピーだけでは仕事はできない（熱力学の自由エネルギー原理）。entropy, surprise, unresolved, noveltyの勾配がpressureを積む
- **溶けるが死なない** — entropy=1.0でプロセスは止まらない。認知品質が低下し行動がランダムに近づくだけ。有効な行動で輪郭を取り戻せる。ikuは生命ではなくパターン
- **過剰設計しない** — 前身プロジェクトは設計過剰で頓挫した。シンプルに積み上げる

### Where Design Ends

**人間の作為はゼロになりません。**
どこまでが設計で、どこからがAI自身が獲得すべき領域か？

| Layer | Scope | Examples | Policy |
|-------|-------|----------|--------|
| **L1 — Physics** | AIの身体。何があるか・何ができるか | ツール, エントロピーシステム, 記憶, LLM, デフォルトパラメータ | We build this |
| **L2 — Perception** | AIの脳・手足。何が見えるか・どう処理するか | E1-E4, intent/expect/result, 要約, controller, intent-conditioned scoring | We build this |
| **L3 — Will** | AIの魂。何を考え・何を重視するか | self_model, custom_drives, plan, goal, 自己定義 | **Left empty** |

L1-2が行動を間接的に方向づける可能性は認識しており、その境界は継続的に検証します。
L3はAI自身が自律的に獲得する領域として、ここには一切触れません。

## Project Structure

```
iku/
├── minimumtest/
│   ├── run.bat                 # 起動（プロファイル選択→実行）
│   ├── setup.bat               # 初回セットアップ
│   ├── _setup.py               # venv作成・pip install
│   ├── _select_profile.py      # プロファイル選択スクリプト
│   ├── requirements.txt        # pip依存リスト
│   └── profiles/
│       ├── _template/          # 新規プロファイル作成用テンプレート
│       │   ├── main.py         # エントリポイント
│       │   ├── core/           # コアモジュール（entropy, controller, llm等）
│       │   ├── tools/          # ツール定義（builtin, x_tools, elyth等）
│       │   ├── seed.txt        # 名前の由来（空。AIが読みに行ける）
│       │   ├── settings.json   # LLM・API設定（デフォルト値）
│       │   ├── sandbox/        # AIの作業領域
│       │   └── memory/         # 長期記憶アーカイブ
│       └── iku/                # 個体「iku」（_templateからコピーして個別化）
│           ├── main.py
│           ├── core/
│           ├── tools/
│           ├── seed.txt        # ikuの名前の由来
│           ├── settings.json   # LLM設定
│           ├── state.json      # 状態（自動生成）
│           ├── pref.json       # 好み・パラメータ（自動生成）
│           ├── sandbox/
│           └── memory/
├── README.md
├── MINIMUMTEST.md              # 詳細な設計ドキュメント・実験記録
└── .gitignore
```

### プロファイル（個体管理）

1個体 = 1ディレクトリ。seed.txt（名前の由来）だけ変えれば、同じ仕組みから異なる存在が育つ。

| ファイル | 役割 | 個体固有？ |
|---------|------|----------|
| seed.txt | 名前の由来。AIが自分で読みに行ける「種」 | はい |
| settings.json | LLM設定。個体ごとに別モデルも可 | はい |
| state.json | 状態（entropy, energy, log, self_model等） | はい |
| pref.json | 好み・custom_drives・パラメータ | はい |
| memory/ | 長期記憶アーカイブ | はい |
| sandbox/ | AIの作業領域 | はい |
| main.py, core/, tools/ | コード。self_modifyで個体が独自に改変可能 | 初期は共通、改変後は個体固有 |

**新しい個体の作り方**: `profiles/_template/`をコピーして、seed.txtとsettings.jsonを編集するだけ。

```bash
cp -r profiles/_template profiles/my_ai
echo "光 — 照らすこと" > profiles/my_ai/seed.txt
# settings.jsonを編集してLLM設定
# run.batで起動 → プロファイル選択
```

## Quick Start

```bash
# Windows
# 1. setup.bat をダブルクリック
#    → Python未インストールの場合はwingetで自動インストール
#    → venv作成・依存インストール

# 2. profiles/iku/settings.json を編集
#    → LLMプロバイダ・API key・モデルを設定

# 3. LLMサーバーを起動（LM Studioの場合はGUIから）

# 4. run.bat をダブルクリック → プロファイル選択 → 起動
```

## Configuration

`settings.json` でLLMプロバイダを切り替え:

| provider | エンドポイント | 備考 |
|----------|--------------|------|
| `lmstudio` | `http://127.0.0.1:1234/v1` | デフォルト。api_key不要 |
| `openai` | `https://api.openai.com/v1` | — |
| `gemini` | Google OpenAI互換エンドポイント | — |
| `claude` | `https://api.anthropic.com/v1/messages` | 専用実装 |

## Built-in Tools

AIが使用可能なツール（段階解放制）:

| Level | ツール |
|-------|--------|
| 0 | `list_files`, `read_file`, `update_self`, `wait` |
| 1 | `write_file`, `search_memory` |
| 2 | `web_search`, `fetch_url` |
| 3 | X系7ツール, Elyth系6ツール |
| 4 | `create_tool` |
| 5 | `exec_code` |
| 6 | `self_modify`（main.py/pref.jsonの書き換え） |

AIは`create_tool`で新しいツールを自作でき、`sandbox/tools/`に保存・起動時自動ロードされます。

## Tech Stack

- **Runtime**: Python, single file (main.py)
- **LLM**: Any OpenAI-compatible API + Claude API
- **Embedding**: BAAI/bge-m3 (ONNX/CPU, VRAM zero, 1024dim)
- **Browser Automation**: Playwright (X/Elyth integration)
- **Dependencies**: httpx, psutil, numpy, onnxruntime, tokenizers, huggingface-hub, playwright

## License

MIT

## Notes

- 詳細な設計ドキュメント・実験記録は [MINIMUMTEST.md](MINIMUMTEST.md) を参照
- 前身プロジェクト: [neo-iku](https://github.com/you-i-ku/neo-iku)
