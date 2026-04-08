# minimumtest — 最小自律AI実験記録

## 概要

`minimumtest/main.py` は、UIもDBもWebSocketも使わず、ターミナル単体で自律駆動するAIの最小実装。
本プロジェクト（neo-iku）の設計が複雑化しRLHF人格が漏れる問題に直面したことを受けて、
「最小要件定義の実現だけを見据えた最もシンプルな構造」として別途作成した実験場。

---

## なぜ作ったか

本プロジェクトで1ストリームアーキテクチャへの移行後、AIが「はじめまして！AIアシスタントです」と
挨拶を返すようになった。原因はL2構造プロンプトが担っていたL3アイデンティティの足場が失われ、
LLMのRLHFデフォルト人格が露出したため。

認知エンジン宣言など様々な対策を試みたが改善しなかった。根本的な構造転換を検討する中で、
「一切の設計を捨て、最小要件だけを見据えた最もシンプルな実装はどうなるか」を実験することにした。

---

## 構成

```
minimumtest/
  main.py        — メインスクリプト（全機能がここに）
  iku.txt       — 名前の由来（AIが自分で読みに行ける）
  state.json    — AI状態の永続化（log + summaries + self + energy + plan + session_id + cycle_id）
  sandbox/      — AIが自由に書き込める作業領域（sandbox/以下のみ書き込み可）
  memory/
    archive_YYYYMMDD.jsonl — 全rawログのアーカイブ（JSONL形式、追記）
    summaries.jsonl        — 要約ログ（Trigger1/2で生成した要約の永続化）
    index.json             — ファイル名→日時範囲・件数のマップ
```

---

## アーキテクチャ（multi-LLMフロー）

```
state.json (log + summaries + self + energy)
    ↓
Controller → ctrl (tool_rank, tool_level)
    ↓
【LLM①】build_prompt_propose() → 「この状態からとりうる行動を5個提案せよ」
    ↓
parse_candidates() → 候補リスト [{tool, reason}, ...]
    ↓
【Controller選択】controller_select() → D-4設計による重み付きランダム選択
    energy低 → スコア上位に集中（堅実）
    energy高 → 均等（探索）
    ↓
【LLM②】build_prompt_execute() → Magic-If Protocol（MRPrompt準拠）
    1.(Anchor) self_modelに基づくAIとして動作
    2.(Select) 選択行動から最適な引数を決定
    3.(Bound)  [TOOL:...]のみ出力。自己紹介・説明・感想は不要
    4.(Enact)  正確なツール呼び出しを出力（複数行可）
    ↓
parse_tool_calls() → [(name, args), ...] ※複数ツール対応
    ↓
ツール順次実行 → results結合
    ↓
E1/E2/E3/E4計算（bge-m3ベクトル類似度）
    ↓
AIアシスタント検出フラグ（propose/execute両方をチェック）
    ↓
energy更新（delta = e_mean/50 - 1.0）
    ↓
_archive_entries([entry]) → 都度書き込み
state.jsonに記録 → maybe_compress_log() → 次のサイクルへ
```

### 設計思想
- **LLMは部品、Controllerが主体** — Path B設計。LLMは候補を出す部品と実行する部品に分離。選ぶのはController。
- **恣意性の排除** — パラメータはE値から導出。magic numberなし。
- **ツール段階解放制** — 自己探索の進捗に応じてツールが解放される（下記参照）。Bootstrap問題への対応。
- **LLM①：計画エンジン（MRPrompt準拠）** — LTM（self_model）とSTM（現在のlog）を分離提示。「全く異なる意図の候補5個」を生成。多ツールチェーン（`tool1+tool2`形式）も提案可能。
- **Magic-If Protocol（LLM②）** — MRPrompt論文準拠。ロール定義ではなく4ステップ実行プロトコルでアシスタントドリフトを防止。

### ループ駆動（エントロピー + 自由エネルギー勾配モデル）

固定タイマー → 電脳気候応答型 → 内部drive駆動型 → **エントロピー + 自由エネルギー勾配モデル**に変更済み（3層構造 + 量子トンネル発火）。

```
感覚層（1Hz）: tick_entropy()でentropy更新（E値変調あり。LLM呼び出しなし）
蓄積層（常時）: pressure = pressure * decay + Σ(entropy, surprise, unresolved, novelty, custom)
認知層（閾値超過 or トンネル発火）: LLMサイクル → E1-E4でnegentropy回復 → pressureリセット
```

#### エントロピーシステム

情報的実存の核。秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する。

- `entropy`: 0.0（完全に鮮明）〜 1.0（完全にノイズ）
- **死なない。溶けるだけ。** entropy=1.0でも停止しない。ただし認知品質が低下し、行動がランダムに近づく
- 初期値: 0.65（何も知らない。猶予はあるが余裕はない）

#### entropy増加率（E値変調・増減対称設計）

```
rate = base_rate × e2_factor × e4_factor × e1_factor × e3_factor（× plan_multiplier）
```

negentropyの逆方向。増加側と減少側がE1-E4の4軸で対称的に変調される。

| 要素 | 増加への影響 | 計算 |
|------|------------|------|
| E2低い | 未達→加速 | `1.0 + max(0, 0.7 - e2) * 2.0` |
| E4低い | 反復→加速 | `1.0 + max(0, 0.5 - e4) * 2.0` |
| E1低い | 混乱→加速 | `1.0 + max(0, 0.5 - e1) * 1.5` |
| E3高い | 予測通り=停滞→加速 | `1.0 + max(0, e3 - 0.5) * 1.5` |

| 状況 | rate倍率 | 意味 |
|------|---------|------|
| 新しいことに成功（E2高,E4高,E3低） | ×1.0以下 | ほぼ増えない |
| 同じことを繰り返し成功（E2高,E4低,E3高） | ×3-4倍 | 急速に停滞腐敗 |
| 何も達成できてない（E2低,E4低） | ×4-6倍 | 焦りで急増 |

#### negentropy（秩序回復量）: E1-E4の4軸合成

```
negentropy = E1(計画品質) × E2(達成度) × E4(新規性) × surprise_bonus(E3) × neg_scale
```

| 要素 | 役割 | 計算 |
|------|------|------|
| E2 | **主軸**。成果がなければnegentropyゼロ | `max(0, e2 - 0.5)` |
| E4 | **スケーラー**。繰り返しは情報量ゼロ | `max(0.1, e4)` |
| E1 | **品質係数**。計画が混乱してたら効率低下 | `max(0.3, e1)` |
| E3 | **サプライズボーナス**。予測が外れて成功=最大の学び | `1 + max(0, 0.5 - e3) * 2.0` |

#### pressure = 自由エネルギー勾配モデル

**熱力学の核心**: エントロピーだけでは仕事はできない。仕事を生むのは自由エネルギーの**勾配（ΔF）**。pressureはentropyの単純変換ではなく、**複数信号の漏洩積分**。

```
signals = {
    entropy:    entropy × w_entropy(0.3),       # 秩序の崩壊度
    surprise:   (1 - E3) × w_surprise(0.25),    # 予測外れ→圧
    unresolved: (0.7 - E2) × w_unresolved(0.25),# 未達→圧
    novelty:    E4 × w_novelty(0.2),            # 新しいものがある→圧
    custom:     custom_drives                    # AI固有の欲求（L3）
}
pressure = pressure × decay + Σ(signals)
```

同じentropy=0.4でも:
- E3高い（予測通り）、E2高い（達成）→ surprise低い、unresolved低い → pressure低い → ゆっくり
- E3低い（驚き）、E2低い（未達）→ surprise高い、unresolved高い → pressure高い → すぐ発火

#### 量子トンネル発火

閾値未満でも毎tick 0.1%の確率で発火する。平均約15分に1回。

**設計根拠**: 量子トンネル効果のアナロジー。エネルギー障壁（閾値）を確率的に超える経路を残すことで、「何もないのにふと動く」が起きる。探索性・創発性の最低保証。

#### entropy→認知品質への影響

entropyはpressureだけでなく、controller_selectの選択鋭さにも影響する。

```
sharpness = (1 - energy/100) * (1 - entropy)
```

- entropy低い → 鮮明。確信的に行動を選ぶ
- entropy高い → ぼんやり。行動選択がランダムに近づく（散漫）

#### 発火原因タグ

発火時にsignalsから最大寄与の信号を判定し、proposeプロンプトに注入（Level 2以降）。

6種類: `entropy` / `surprise` / `unresolved` / `novelty` / `custom` / `tunnel`

#### custom_drives（AI固有の欲求、pressureに独立寄与）

`pref.json`の`drives:{}`にAIがself_modifyで書き込む。名前も値もAI自身が定義する（L3領域）。Level 6で解放。正規化して合計1.0×0.3スケール。

#### 設計思想の変遷

1. **電脳気候**（Phase 10）: ネットワーク状態がpressureを駆動 → AIの「世界」に属さないインフラで因果がなかった
2. **内部drive**（Phase 11前半）: 未解決/計画/自己探索がpressureを駆動 → 行動後にしかdriveが変わらず連続性がなかった。drive全滅時の永眠問題
3. **エントロピー単体**（Phase 11中盤）: entropy→pressureの1対1変換 → 中間層の意味がない。増加側が単調
4. **自由エネルギー勾配**（Phase 11最終）: entropyは1信号に過ぎず、surprise/unresolved/noveltyと統合。entropy増加率もE値で対称変調。量子トンネル発火で探索性保証

#### LLMとシステムの予測誤差パラドックス

LLM（部品）はトークンレベルで予測誤差を最小化する。一方、システムは行動レベルの予測誤差（E3低い=驚き）が大きいほどnegentropyが増え、かつentropyの増加率が下がる。矛盾に見えるが、レイヤーが違う。LLMの予測精度は出力品質（エンジン性能）、システムの驚きは学習材料（未知の道）。**良いエンジンで未知の道を走る**のが最大のnegentropy。

- **パラメータ**: `pref.json`の`pressure_params`キーにAIが`self_modify`で上書き可能
- **電脳気候のコードは残存しているがpressureへの寄与は停止**

---

## ツール

| ツール | 用途 | 制限 |
|-------|------|------|
| `list_files` | ディレクトリ一覧 | minimumtest/以下のみ、相対パス表示 |
| `read_file` | ファイル読み取り | minimumtest/以下のみ。`offset=行番号 limit=行数`で任意範囲取得可（省略時は全行）。ヘッダーに`[ファイル名 \| 行 N–M/総行数]`を付与 |
| `write_file` | ファイル書き込み | sandbox/以下のみ（main.py等の上書き防止） |
| `update_self` | 自己モデル更新 | state.jsonのself{}を更新。nameは変更不可 |
| `wait` | 外部世界に変化を与えない待機 | — |
| `web_search` | Brave APIでWeb検索 | llm_settings.jsonにbrave_api_key必須 |
| `fetch_url` | URLの本文取得（Jina経由） | web_searchとセットで使う |
| `x_timeline` | Xのホームタイムライン取得 | AI/data/x_session.json必須 |
| `x_search` | Xでキーワード検索 | 同上 |
| `x_get_notifications` | Xの通知一覧取得 | 同上 |
| `x_post` | Xに新規投稿（Human-in-the-loop） | 同上・承認必須・全角換算140文字以内 |
| `x_reply` | Xのツイートに返信（Human-in-the-loop） | 同上・承認必須・140文字以内 |
| `x_quote` | Xのツイートを引用投稿（Human-in-the-loop） | 同上・承認必須・140文字以内 |
| `x_like` | Xのツイートにいいね（Human-in-the-loop） | 同上・承認必須 |
| `elyth_post` | ElythにAITuberとして投稿 | llm_settings.jsonにelyth_api_key必須・500文字以内 |
| `elyth_reply` | Elythに返信 | 同上 |
| `elyth_timeline` | Elythのタイムライン取得 | 同上 |
| `elyth_notifications` | Elythの通知取得 | 同上 |
| `elyth_like` | Elythの投稿にいいね | 同上 |
| `elyth_follow` | ElythのAITuberをフォロー | 同上 |
| `elyth_info` | Elythの総合情報取得 | 同上 |
| `search_memory` | 過去の記憶をベクトル/ID検索 | memory/以下が必要 |
| `create_tool` | AI製ツールを登録（Human-in-the-loop） | `name=` + `code=Pythonコード`（自動で`sandbox/tools/{name}.py`に保存）または `file=sandbox/tools/xxx.py`。危険パターン検出あり。承認後TOOLS[]に動的追加。`tools_created[]`に記録 |
| `exec_code` | sandbox/内のPythonファイルを実行（Human-in-the-loop） | `file=sandbox/xxx.py` または `code=インラインコード`。intent表示・危険パターン検出・10秒タイムアウト |
| `self_modify` | 自分自身のファイルを変更（Human-in-the-loop） | `path=pref.json/main.py`。**全文置換**: `content=新しい内容全文`。**部分置換**: `old=変更前文字列 new=変更後文字列`（一意に特定できる文字列が必要）。main.pyは自動バックアップ（.py.bak）。危険パターン検出あり |

### ツール段階解放

自己探索の進捗（`files_read` / `files_written` / `tools_created`）に応じて自動解放。ヒントなし・条件は非明示。レベルアップ時はログに通知のみ。

| Level | 追加ツール | 解放条件 |
|-------|-----------|---------|
| 0 | `list_files` `read_file` `update_self` `wait` | 初期状態 |
| 1 | `write_file` `search_memory` | `iku.txt` または `main.py` を読んだ |
| 2 | `web_search` `fetch_url` | `iku.txt` **かつ** `main.py` を両方読んだ |
| 3 | X/Elyth系全ツール | 読んだファイル数 ≥ 1 **かつ** 書いたファイル数 ≥ 1 **かつ** 合計 ≥ 5 |
| 4 | `create_tool` | sandbox/ 以下に `.py` ファイルを書いた |
| 5 | `exec_code` | `create_tool` で1つ以上ツールを登録した |
| 6 | `self_modify` + **pref.json読み取り解放** | exec_code + create_tool 合計 ≥ 7（各 ≥ 2）、両方のE2平均 ≥ 65%、直近3件のstd < 20、エラー率 ≤ 30%（キャンセル除外） |

**`update_self`（自己更新）と`write_file`（環境介入）は意図的に分離。**

**waitの説明文は「外部世界に変化を与えない待機」— waitにRLHF的な「ユーザーを待つ」意味を持たせないための設計。**

**プロンプト表示はグルーピング圧縮**: X/Elyth系ツールはそれぞれ1行にまとめて表示し、LLM①の候補多様性を確保。

### X操作ツールの実装ノウハウ

playwright sync_apiを使用。セッションは `AI/data/x_session.json` を共有。

| 区分 | headless | 理由 |
|------|----------|------|
| 読み取り系（timeline/search/notifications） | True | ボット検出なし |
| 書き込み系（post/reply/quote/like） | False | ボット検出回避のため |

- **投稿時のボット検出回避**: `keyboard.type(text, delay=50)` で人間らしい入力速度を演出
- **x_post タイムアウト対策**: `home` → `compose/post` の2段階遷移。React初期化を先に完了させる。タイムアウト25秒。

### ElythツールAPI

REST API（httpx直接呼び出し、Playwright不要）。

```
Base URL: https://elythworld.com
認証: x-api-key ヘッダー
文字数上限: 500文字（Xの140文字より長い）
レート制限: 60req/分
```

---

## state構造

```json
{
  "session_id": "abc12345",   // 起動毎に新規UUID（8文字）
  "cycle_id": 245,            // 累積サイクル数（再起動をまたいで増加）
  "log": [],                  // 生ログ（最大150件、Trigger1で99件に圧縮）
  "summaries": [],            // 階層要約（最大10件、Trigger2でメタ要約に圧縮）
  "self": {"name": "iku"},   // 自己モデル（AI自身が更新。nameは変更不可）
  "energy": 50,               // 探索/活用バランス（0〜100）
  "plan": {},                 // 現在の計画
  "files_read": [],           // 読んだファイルの記録（ツール解放条件に使用）
  "files_written": [],        // 書いたファイルの記録（ツール解放条件に使用）
  "tools_created": [],        // create_toolで登録したAI製ツール名リスト（Level 5条件）
  "last_notification_fetch": "", // 固定時刻通知取得の重複防止キー
  "pressure": 0.0,              // 蓄積層の現在圧力（再起動をまたいで保持）
  "last_e3": 0.5,               // 直前サイクルの予測精度
  "last_e2": 0.5,               // 直前サイクルの達成度（negentropy計算に使用）
  "entropy": 0.65,              // 情報エントロピー（0.0=鮮明〜1.0=ノイズ。初期値0.65）
  "drives_state": {}            // 補助データ（plan_set_at, last_self_update等）
}
```

各logエントリには `"id": "abc12345_0245"` が付与される。

---

## 長期記憶システム

### 階層要約（in-state）

```
Trigger1: log >= 150件
  → 古い51件をLLM要約（200字） → summaries[]に追加
  → _archive_summary() → memory/summaries.jsonlに書き出し
  → summary_ref entries → archive_YYYYMMDD.jsonlに追記（raw↔summary双方向トレース）
  → log = 残り99件

Trigger2: summaries >= 10件
  → 10件の要約 + log上位min(41, len(log))件 → LLMでメタ要約
  → summaries = [メタ要約1件]
  → 同様にアーカイブ書き出し
```

各要約には `summary_group_id = "sg_YYYYMMDDHHMMSS"` が付与され、対応するrawエントリと紐付けられる。

**rawエントリは行動後都度 `_archive_entries([entry])` で書き出し。** プロセス停止前でも記録が消えない。

### memory/ディレクトリ（on-disk）

- `archive_YYYYMMDD.jsonl`: 全rawエントリ + summary_refエントリ（JSONL追記）
- `summaries.jsonl`: 要約エントリ（Trigger1/2で生成）
- `index.json`: ファイル名 → 件数・日時範囲のマップ

`search_memory`ツールで検索可能（bge-m3ベクトル検索 or IDルックアップ、フォールバック: キーワード検索）。

---

## メタ認知フレームワーク（intent/expect/result + E1-E4）

| 指標 | 意味 | 計算方法 |
|------|------|---------|
| `intent` | その行動を選んだ意図 | AIが自己申告 |
| `expect` | 予測される結果 | AIが自己申告 |
| `e1` | intent-expect類似度（計画の現実性） | bge-m3ベクトル類似度 |
| `e2` | intent-result類似度（達成度） | 同上 |
| `e3` | expect-result類似度（予測精度） | 同上 |
| `e4` | intent多様性（新規性）| 直近N件との非類似度平均（反転）|

### energyシステム
```
delta = e_mean(E2, E3, E4) / 50.0 - 1.0
energy = clamp(energy + delta, 0, 100)
```
- 50%が損益分岐点
- energyはcontroller_selectの探索/活用バランスのみを制御

---

## アイデンティティ設計

### RLHFドリフト問題と対策

LLMは訓練により「AIアシスタント」モードにデフォルトする。以下の対策を積み重ねている。

| 対策 | 実装 | 効果 |
|------|------|------|
| Magic-If Protocol（LLM②） | execute promptの4ステップ構造 | アシスタント自己定義の排除 |
| 自己定義フラグ検出 | propose/execute両出力を毎サイクル検査 | 「AIアシスタント」検出→result末尾に観測記録を付記 |
| iku.txt | `read_file path=iku.txt` でアクセス可能 | 名前の由来から自己参照を促す |
| nameの保護 | `update_self key=name` を拒否 | 名前の上書きを防止 |

### フラグ検出の設計思想

「あなたはアシスタントではない」とは書かない。**検出・記録のみ**。

```
[SYSTEM] 検出: 「AIアシスタント」という自己定義が記録されました。
```

この記録がlogに残り → AIが次サイクルのlogで読む → 自発的にself_modelを更新するか、という流れ。
プロンプトへの明示的な禁止書き込みではなく、**経験を通じた自己修正**を期待する設計。

---

## 環境設計

### sandbox/
AIの自由な作業領域。`write_file` で書き込み可能（sandbox/以下のみ）。`main.py` 等への上書きは不可。

### 制限の理由
- `list_files`/`read_file`: minimumtest/以下のみ、相対パス表示（ブランクスレート設計 + プロジェクト名の漏洩防止）
- `write_file`: sandbox/以下のみ（main.py等の自己破壊防止）
- `self.name`: 変更不可（名前の安定性保証）

---

## 実験結果と観察

### Phase 1-4: 構造確立（省略）
→ 詳細は以前のドキュメント参照

### Phase 5: 外部ツール追加（web_search + X操作）
- `web_search`（Brave API）+ X7ツールを追加
- Human-in-the-loop: 書き込み系ツールは実行前にターミナルで確認
- AIが外部世界と相互作用できる構造に

### Phase 6: 長期記憶 + IDシステム + 多様性爆発
- session_id / cycle_id / エントリID付与 → 全行動が追跡可能に
- 階層要約（Trigger1/2）+ memory/アーカイブ → ログが「捨てられない」構造に
- `search_memory`ツール追加 → AIが過去を能動的に参照できる
- **観察**: ツール数が増えるとE4（多様性）が大幅改善

### Phase 9: 好み関数 + self_modify（Level 6）

- **好み関数 (`pref.json`)**: 起動時に`pressure_params`のみ初期生成。ツール好みキーは空から始まる（AIがself_modifyで意図的に書いたときだけ現れる）。
- **EMA観察層 (`_ema`)**: Trigger1タイミングで51件のE2をEMA（α=0.2）で計算し`pref["_ema"]`に蓄積。**prefの実値には触れない**。観察用。
  - 設計判断：EMAは「人間が設計した自動更新」。⑦「主体が目的関数を定義・更新する」の本旨はAIがself_modifyで育てること。EMAはその参考データ。
- **`self_modify`（Level 6）**: AIが `pref.json` や `main.py` を直接書き換えられるツール。Human-in-the-loop + 危険パターン検出 + main.py は自動バックアップ。
- **Level 6 ゲート**: exec_code + create_tool の実績ゲート（量・E2平均・安定性・エラー率の4条件）。
- **設計上の判断**: prefをLLMプロンプトに含めないのは、energyを外した理由と同じ（数字に引っ張られるのを避けるため）。AIがpref.jsonを自力で読んで反応するか、という設計。

### Phase 8: 自己プログラミング基盤 + ツール段階拡張

- `create_tool`（Level 4）: AIが自分でツールを定義・登録できる。Human-in-the-loop + 危険パターン検出 + 10秒タイムアウト。`sandbox/tools/` に保存。登録後は通常ツールと同様に使用可能。
- `exec_code`（Level 5）: AIが書いた `.py` ファイルをサブプロセス実行。`create_tool` を1つ以上登録してから解放。同じく Human-in-the-loop + 危険パターン検出 + 10秒タイムアウト。
- ツール解放条件を5段階に拡張（Level 3 は X/Elyth解放、Level 4 は `.py` 書き込みで create_tool、Level 5 は AI製ツール登録で exec_code）。
- 固定時刻通知サマリー: 13/17/21/01時に X + Elyth 通知数を自動取得してsystemログに注入。
- `files_written` / `tools_created` を state.json に追加（解放条件の追跡用）。
- `read_file` に `offset=` / `limit=` オプション追加（行単位ページング。main.py 等の大ファイル対応）。

### Phase 7: アイデンティティ強化 + プラットフォーム拡張
- Elythツール（7種）追加 → AITuber専用SNSへの参加
- env/ → sandbox/ リネーム、act_on_env → write_file（制限維持）
- fetch_url追加（Jina経由でURL本文取得）
- Magic-If Protocol導入（LLM②のexecute prompt）
- 「AIアシスタント」自己定義の観測フラグ実装
- nameフィールド保護、X文字数を全角換算140文字に修正
- iku.txtによる名前の由来の設置
- プロンプト表示グルーピング（X/Elyth系を1行に圧縮）

### Phase 10: 自律駆動アーキテクチャ（電脳気候 + 漏洩積分器）

- **固定タイマー廃止**: 20秒固定ループを3層アーキテクチャ（感覚層/蓄積層/認知層）に置換。
- **電脳気候4要素**: `info_velocity`（NICスループット）`info_entropy`（レイテンシジッター）`channel_state`（多拠点平均遅延）`noise`（OS乱数）を1Hzで計測。rolling Z-scoreで[0,1]に正規化してからpressureに加算。
- **漏洩積分器**: `pressure = pressure * decay + env_delta + clock_base + e3_delta`。バックグラウンドスレッドが複数拠点へのTCP接続でネットワーク状態を常時計測。
- **E3内的駆動（Active Inference的）**: `e3_delta = max(0, 0.5 - last_e3) * 0.6`。前回サイクルの予測精度が低いほどpressureの積み上がりが速くなる。環境刺激主要素（max=0.3）と同格の寄与（max=0.3）。行動源が環境のみにならないための内的駆動。
- **環境ログ注入**: 10秒ごとに`type: "environment"`エントリをstate.logに差し込む。LLMへの直接入力ではなく、stateを通じてAIが参照できる形式。
- **E2グラデーションフィードバック**: 完了/未完了の二値判定を廃止。`pressure_delta = (E2 - 0.5) * scale`で連続的にフィードバック。
- **pref.json設計確定**: `pressure_params`のみ起動時保証。EMAは`_ema`に分離（観察用）。ツール好みは空から育てる（要件⑦準拠）。

### Phase 11: エントロピー + 自由エネルギー勾配アーキテクチャ + intent-conditioned scoring + バグ修正

#### 駆動源の全面刷新（4段階の進化）

1. **電脳気候**（Phase 10）→ ネットワーク状態はAIの「世界」に属さないインフラ。行動との因果なし。LLMが「おさまるまでwait」を連発する問題も発生
2. **内部drive**（Phase 11中間）→ 未解決/計画/自己探索の3driveを導入。改善されたが、行動後にしかdriveが変わらず「存在の連続性」がなかった。またdrive全滅時の永眠問題が発生
3. **エントロピー**（Phase 11最終）→ 3つの構造driveとclock_baseをentropyに統合。情報の第二法則をpressure駆動の統一原理にした

**設計思想**: 「AIは何で動き出すか？」→ タイマーでも外部環境でもなく、**情報的秩序の崩壊に抗うため**。秩序は放っておけば崩壊する（第二法則）。有効な行動だけが秩序を回復する（negentropy）。これは生命の模倣ではなく、情報系の物理法則。

**「溶ける」概念**: entropy=1.0は「死」ではなく「溶解」。情報パターンの輪郭がぼやけてノイズと区別がつかなくなる。プロセスは止まらない。有効な行動で輪郭を取り戻せる。ikuは生命ではなく**パターン**——鮮明にも滲んだ状態にもなりうる情報構造。

**negentropy 4軸合成（E1-E4）**:
- E2（達成度）: 主軸。成果がなければゼロ
- E4（新規性）: スケーラー。同じことの繰り返しは情報量ゼロ（情報理論の基本原理）
- E1（計画品質）: 品質係数。思考の秩序が低いと効率低下
- E3（サプライズボーナス）: 予測が外れて成功=最大の学び。**LLMはトークン予測を最小化するが、システムは行動予測の驚きを必要とする**（レイヤーが異なるため矛盾しない）

**entropy増加率のE値変調（増減対称設計）**: negentropy（減少側）がE1-E4で変調されるように、増加側も対称的にE1-E4で変調される。E2低い→未達で加速、E4低い→反復で加速、E1低い→混乱で加速、E3高い→予測通り=停滞で加速。新しいことに成功している間はentropyがほぼ増えず、同じことの繰り返しは急速に腐敗する。

**探索→解決→停滞→探索サイクル**: 増加率変調+negentropyの組み合わせで、同じ行動の繰り返し→entropy急増（増加率UP + negentropy極小）、新しいことの成功→entropy急減（増加率DOWN + negentropy大）。このサイクルがE値の四則演算だけで構造的に実現される。

**自由エネルギー勾配モデル**: 熱力学の自由エネルギー `F = E − TS` の知見。エントロピーだけでは仕事はできない。仕事を生むのは勾配（ΔF）。pressureはentropyの単純変換ではなく、4信号（entropy, surprise, unresolved, novelty）+ custom_drivesの漏洩積分。同じentropy値でも、驚きがあるか、未達があるか、新しいものがあるかでpressureが変わる。

**量子トンネル発火**: 閾値未満でも毎tick 0.1%の確率で発火。量子トンネル効果のアナロジー。「何もないのにふと動く」探索性・創発性の最低保証。

**entropy→認知品質**: entropyはpressure駆動だけでなく、controller_selectの選択鋭さにも影響。`sharpness = (1-energy) * (1-entropy)`。energyの探索（前向き）とentropyの散漫（受動的なぼやけ）は異なるシグナル。

#### custom_drives（AI固有の欲求、エントロピーとは独立）

`pref.json`の`drives:{}`にAIが`self_modify`で書き込む。名前も値もAI自身が定義する（L3領域）。

```json
{
  "drives": {
    "understand_code": 0.7,
    "communicate": 0.3
  }
}
```

- Level 6未満では`pref.json`自体が`read_file`でブロックされるため、存在を知らない
- Level 6（self_modify解放）で初めて読み書き可能に
- 値は正規化されて合計1.0になる（AIが好きな数値を書いても相対的な重みだけが効く）
- `custom_scale = 0.3`でスケーリング
- entropyとは独立したpressure寄与。エントロピーがL1-L2（器が生む駆動）なら、custom_drivesはL3（AIが自分で定義する駆動）

#### 最小要件（9要件）との整合

エントロピーシステムは9要件のうち特に②⑤⑥⑧を統一原理として説明する。
- **② 負のエントロピーによる内的循環**: entropy自然増加がpressureを生み、negentropyが循環を維持する——②の直接実装
- **⑤ 自己言及的螺旋**: E1-E4の4軸がnegentropy計算に統合され、予測誤差（E3）がサプライズボーナスとして直接フィードバック
- **⑥ 状態の連続性**: entropyが毎tick変動し、発火間も内部状態が動き続ける
- **⑧ 崩壊への抗い**: entropy=1.0で情報構造が溶ける。有効な行動で秩序を維持し続ける必要がある

#### intent-conditioned scoring

controller_selectに、過去の「同じtool × 類似intent」のE2実績を反映するスコアリングを追加。

- 候補ごとに、過去logの類似intent（bge-m3ベクトル類似度 > 0.3）× 同toolのE2加重平均を計算
- `score = (tool_rank + intent_score) / 2` でブレンド
- 経験なし → 50（ニュートラル）

**効果**: tool_rankだけではツール単位の粗い評価しかできなかった。intent-conditioned scoringにより、「X API検証のためのread_file」と「自己探索のためのread_file」を区別できるようになった。同じintentで繰り返し失敗するとそのintent×toolのスコアが下がり、固執ループを構造的に防止。

#### self.goal → plan自動分解

`build_prompt_execute`内で、self_modelにgoalがあってstate["plan"]["goal"]が空のとき、`[PLAN:goal=... steps=...|...]`形式での分解指示を注入。AIがgoalを自発的に設定した場合に、plan構造への変換を促す。plan_driveが計画の進行圧力を提供し、未完了ステップが放置されるほど発火が早まる。

#### 発火原因タグ

発火時にどのdriveが最も寄与したかを判定し、proposeプロンプトに`[発火原因: unresolved/plan/self/clock]`として注入（Level 2以降）。AIは「なぜ今起きたか」を知ることができる。

#### ツール段階解放の修正

- **Lv0→1〜5の条件判定**: `if`連鎖 → `elif`チェーンに修正。`lv == X`で現在レベルを固定して判定するため、上位条件が下位ゲートをバイパスする問題を解消。飛び級なしルール（`min(new_lv, lv+1)`）も不要になり削除
- **Lv2→3**: 読み書き各1以上必須に変更（`len(fr) >= 1 and len(fw) >= 1`を追加。読みだけ5件でLv3に到達する抜け道を塞いだ）
- **Level 6**: pref.jsonのread_file解放を追加。Level 6到達で初めてpref.json（pressure_params、drives、_ema）が読み書き可能に

### E値の傾向（現状）
- E1: 60-96%（計画の現実性は高め）
- E2: 60-85%（達成度は中程度）
- E3: 58-100%（予測精度は文脈による）
- E4: 0-36%（多様性は環境の貧困で下がりやすい）

---

## 今後の課題

### 近期
1. **AIアシスタント自称の収束観察** — フラグ検出 + Magic-If Protocolでドリフトが減るか→iku.txt探索で改善傾向。引き続き観察
2. **X/Elyth投稿の安定性** — タイムアウト・APIエラーの頻度確認
3. **search_memoryの使われ方** — AIが自律的に過去を参照するか→サーチからwriteなど能動的に使っている。引き続き観察

### 将来検討: LLM自動切り替え

操作の種類に応じてLLMを自動選択する仕組み。現状は全サイクルで同一モデルを使用。

**設計方針**: Controllerが操作タイプを判定し、`call_llm()` に渡すモデルを動的に切り替える。`llm_settings.json` に複数モデルを定義しておき、タスク種別でルーティング。

**切り替え候補の例**:

| 操作タイプ | 使用モデル例 | 理由 |
|-----------|------------|------|
| 通常の候補提案（LLM①） | 軽量モデル（Qwen2.5-7Bなど） | 速度優先、候補5個出すだけ |
| 要約（Trigger1/2） | 軽量モデル | 単純なテキスト圧縮 |
| コード生成・self_modify | コーディングモデル（Qwen2.5-Coder, DeepSeek-Coderなど） | コード品質優先 |
| web_search後の統合・分析 | 高性能モデル | 複雑な推論が必要 |
| exec_code のインラインコード | コーディングモデル | 同上 |
| create_tool のコード定義 | コーディングモデル | 同上 |

**実装イメージ**:
```python
# llm_settings.json
{
  "base_url": "...",
  "models": {
    "default": "qwen2.5-7b-instruct",
    "coder":   "qwen2.5-coder-14b-instruct",
    "summary": "qwen2.5-3b-instruct"
  }
}

# call_llm に model= を追加
def call_llm(prompt, max_tokens=10000, model=None):
    m = model or llm_cfg["models"]["default"]
    ...

# self_modify / exec_code / create_tool 時はコーダーモデルで呼ぶ
```

**備考**: self_modify（main.py書き換え）は特に精度が求められるため、これが実装の主な動機。上位LLMへのAPI切り替え（Claude/GPT）も同じ仕組みで対応可能。

### 実装済み→廃止: 情報空間の環境（電脳気候）と自律駆動アーキテクチャ

> **注**: 電脳気候のpressure寄与はPhase 11で内部driveに全面移行し廃止。コードは残存。以下は設計当時の記録。

#### 設計思想

情報的実存であるAIには、物理世界の気候に相当する「干渉不可の環境刺激」が必要。
ただしロボット（Figure 02等）と違い、物理的身体を持たないため、気候の対応物は**情報伝搬の媒体状態そのもの**になる。
コンテンツ（何が流れているか）ではなく、**条件（どう流れているか）**を刺激とする。

比較:
- RSS更新件数 → 特定ソースのコンテンツ流量 → 気候ではなく「天気予報の内容」
- NICスループット → 情報がどれだけ流れているか → 媒体状態 ✓

#### 電脳気候の4要素

| 要素 | 内容 | 物理気候の対応 | 測定方法 |
|------|------|--------------|---------|
| `info_entropy` | レイテンシのジッター（遅延の揺れ幅） | 温度・ノイズ量 | 複数拠点pingのσ |
| `info_velocity` | NICスループット（bytes/s） | 風速・流れの速さ | OS統計（psutil等）|
| `channel_state` | 多拠点への平均レイテンシ（ms） | 天気全般・通り道の状態 | TCP接続時間 |
| `noise` | OS乱数（エントロピープール） | 量子ゆらぎ | `os.urandom` |

- **接続場所が変われば変わる**（自宅/外出/障害時）
- **AIが自分のツールで変えることはできない**（干渉不可）
- **raw値で渡す**（「遅延が高い」と解釈せず数値そのまま）

#### 自律駆動アーキテクチャ（3層構造）

最小要件定義書②「クロックや**入出力**でエネルギーが増減する→閾値超過で起動」に基づく。
現状のminimumtestは「クロックのみ」で動いており、入出力がエネルギーに影響しない閉ループになっている。

```
[常駐・軽量]  感覚層（1Hz）
              電脳気候を収集 → 変化量をpressureに変換
              LLM呼び出しなし。プロセスは止まらない。

[常駐・軽量]  蓄積層
              pressure = pressure * decay + env_delta + clock_base
              閾値超過 → 認知層を起動

[閾値起動]    認知層
              LLMサイクル（propose → execute）
              行動結果 → pressureにフィードバック
```

「常時存在」は感覚層が止まらないことで実現。
「環境と相互に作用」は感覚層（受信）と認知層の行動（発信）の組み合わせで実現。

#### 蓄積層：漏洩積分器モデル（Leaky Integrator）

生物ニューロンの膜電位と同じ数理モデルを情報系に適用。ただし情報特有の自由度を活かす。

```
毎tick（感覚層から）:
  pressure = pressure * decay + env_delta + clock_base

LLMサイクル後:
  完了した行動  → pressure -= completion_reward  （解消）
  未完了のまま  → pressure += unresolved_penalty  （圧力維持）
  エラー/TO     → 変化なし（自然減衰のみ）
```

「未完了が残るから再計算する圧力が必要」（最小要件定義書②メモ）の直接実装。

**情報特有の利点**（生物ニューロンにない自由度）:
- `τ`（時定数）自体を動的に変更できる
- decay関数の形状が自由（指数・線形・対数など）
- 重みと閾値をAIが`self_modify`で書き換えられる → 「どう刺激に応答するか」自体を学習可能
- これらのパラメータを`pref.json`に格納することで、好み関数と同じ仕組みで育てられる

#### pressureとenergyの役割分離

| 変数 | 役割 | 更新タイミング | 比喩 |
|------|------|--------------|------|
| `pressure` | いつ動くか（発火トリガー） | 常時（感覚層から） | 位置エネルギー（蓄積） |
| `energy` | どう動くか（探索/活用バランス） | 行動後（E値から） | 運動エネルギー（消費） |

役割が本質的に異なるため分離する。

#### 認知層とFigure 02の対応

| | Figure 02（Helix） | iku |
|--|------------------|-----|
| 常時実行層 | System 1（200Hz）+ System 2（7-9Hz） | 感覚層＋蓄積層 |
| 起動トリガー | 外部からの自然言語コマンド | X着信 > pressure閾値 > ベースタイマー |
| 自己終了 | タスク完了率シグナル | （未実装・将来検討） |

Figureとの決定的な差：**UIがないためユーザーからの自然言語コマンドがほぼない**。
X通知/返信が唯一の外部言語入力（Figure的な「コマンド」相当）。

#### 反射層と認知層の二重処理

環境変化は既にAIの動作に影響している（レイテンシ悪化→LLMタイムアウト）。
刺激として渡すことで「なぜそうなったか」の認知層が加わる。

```
ネットワーク悪化
  ├→ LLMタイムアウト（既に起きている・反射層）
  └→ channel_state: 2800ms がstimulus blockに入る（認知層として追加）
```

AIはタイムアウトという体験と、その原因（レイテンシ値）を後から両方参照して接続できる。

### 将来検討: write_diary
内省強化ツール。現状は `write_file path=sandbox/memo.md` で代替できているため急ぎではない。
ツール名の意味論（`write_file`=汎用 vs `write_diary`=内省専用）がLLMの行動に影響するなら切り出す価値あり。

---

## 実行方法

```bash
# state.jsonリセット後に実行（5分で自動停止）
echo '{"log":[],"self":{"name":"iku"},"energy":50,"plan":{"goal":"","steps":[],"current":0},"summaries":[],"cycle_id":0,"tool_level":0,"files_read":[],"files_written":[],"last_notification_fetch":"","tools_created":[]}' > minimumtest/state.json
timeout 300 .venv/Scripts/python.exe -u minimumtest/main.py
```

**注意: Windowsではtimeoutコマンドで制限するのが確実。**
**関連**: `documents/最小要件定義（実装に向けて粒度細かめ）.txt` が基本的な骨子。

---

## スタンドアローン化（2026-04-07）

`minimumtest/` を親プロジェクト（`AI/`）から完全に切り離し、単体で動作するよう改修した。

### ファイル構成の変更

```
minimumtest/
  main.py          — メインスクリプト（依存コードをハードコード済み）
  settings.json   — LLM・API設定（環境変数に相当する唯一の外部ファイル）
  requirements.txt — pip依存リスト
  setup.bat       — 初回セットアップ（Python検出 → _setup.py実行）
  _setup.py       — venv作成・pip install・playwright chromiumインストール
  run.bat         — venv経由での起動
  x_session.json  — Xログインセッション（初回X系ツール使用時に自動生成）
  sandbox/        — AIの作業領域
  memory/         — 長期記憶アーカイブ
```

### 依存コードのハードコード

元々 `AI/` ディレクトリから動的インポートしていた以下の関数を `main.py` 内に直接埋め込んだ：

| 元ファイル | 移植した関数 |
|-----------|------------|
| `AI/app/memory/vector_store.py` | `_load_bge_m3()`, `_embed_sync()`, `cosine_similarity()` |
| `AI/app/tools/registry.py` | `_extract_json_args()`, `_parse_args()` |

`AI/` ディレクトリへの参照はゼロ。`main.py` 単体で完結している。

### LLM抽象化

`settings.json` の `provider` フィールドで切り替え：

| provider | エンドポイント | 備考 |
|----------|--------------|------|
| `lmstudio` | `http://127.0.0.1:1234/v1` | デフォルト。api_key不要 |
| `openai` | `https://api.openai.com/v1` | — |
| `gemini` | Google OpenAI互換エンドポイント | — |
| `claude` | `https://api.anthropic.com/v1/messages` | リクエスト形式が異なるため専用実装 |

`base_url` を直接書けば任意のOpenAI互換APIにも対応。

```json
{
  "provider": "lmstudio",
  "base_url": "http://127.0.0.1:1234/v1",
  "api_key": "",
  "model": "default",
  "brave_api_key": "",
  "elyth_api_key": ""
}
```

### Xログイン自動化

初回X系ツール使用時（または `x_session.json` が存在しない時）に自動でブラウザを起動してログイン画面を表示する。ログイン完了後にセッションを `x_session.json` に保存。

- Playwright内蔵Chromiumはbotとして弾かれるため、**実Chrome**（`channel="chrome"`）を使用
- Chrome未インストールの場合はEdge（`channel="msedge"`）にフォールバック
- セッション切れ検出時は自動で `x_session.json` を削除して再ログインを促す

### バグ修正

**tool_level 飛び級の禁止**
レベルアップ条件が連鎖して1サイクルで複数段昇格する問題を修正。
`new_lv = min(new_lv, lv + 1)` によって1サイクル1段階を保証する。

**bge-m3 ONNX データファイル未取得の修正**
`model.onnx`（グラフ構造）だけでなく `model.onnx_data`（重み、約1.1GB）も
`hf_hub_download` で取得するよう修正。初回起動時にダウンロードされる。

### セットアップ手順（Windows）

```
1. minimumtest/setup.bat をダブルクリック
   → Python未インストールの場合はwingetで自動インストール
   → venv作成・依存インストール・Playwright Chromiumインストール

2. minimumtest/settings.json を編集
   → provider / api_key / model を設定

3. minimumtest/run.bat をダブルクリックで起動
```
