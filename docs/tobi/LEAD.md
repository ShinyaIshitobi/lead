# LEAD: Minimizing Learner–Expert Asymmetry in End-to-End Driving 解説書

> 論文: [arXiv:2512.20563](https://arxiv.org/abs/2512.20563)
> 著者: Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner, Maximilian Igl, Andreas Geiger, Kashyap Chitta
> 学会: CVPR 2026 accepted

---

## 目次

1. [はじめに: この論文が解決する問題](#1-はじめに-この論文が解決する問題)
2. [前提知識: E2E自動運転の基礎](#2-前提知識-e2e自動運転の基礎)
3. [核心: Learner–Expert Asymmetry（学習者とエキスパートの非対称性）](#3-核心-learnerexpert-asymmetry学習者とエキスパートの非対称性)
4. [手法1: State Alignment（状態の整合）](#4-手法1-state-alignment状態の整合)
5. [手法2: Intent Alignment（意図の整合）](#5-手法2-intent-alignment意図の整合)
6. [TFv6: TransFuser V6 アーキテクチャ](#6-tfv6-transfuser-v6-アーキテクチャ)
7. [学習パイプライン](#7-学習パイプライン)
8. [実験結果](#8-実験結果)
9. [実世界への展開: NAVSIM / Waymo](#9-実世界への展開-navsim--waymo)
10. [コードベースとの対応](#10-コードベースとの対応)
11. [限界と今後の課題](#11-限界と今後の課題)
12. [用語集](#12-用語集)
13. [背景技術と関連論文ガイド](#13-背景技術と関連論文ガイド)
14. [エキスパートの4段階パイプライン](#14-エキスパートの4段階パイプライン)
15. [Known Issues と実践Tips](#15-known-issues-と実践tips)

---

## 1. はじめに: この論文が解決する問題

自動運転シミュレータ（CARLAなど）では無限にデータを生成できる。にもかかわらず、模倣学習（Imitation Learning）で訓練されたポリシーは、実際の閉ループ走行（closed-loop driving）で安定した性能を出せない。なぜか？

この論文の答えは明快で、「教師（Expert）と生徒（Learner）の間に、根本的な非対称性がある」という点に尽きる。

教師はシミュレータの特権情報（全車両の位置・速度、隠れたオブジェクトの情報等）にアクセスできるが、生徒はカメラやLiDARなどのセンサ情報しか使えない。この非対称性を無視したまま「完璧な教師の真似をしろ」と訓練しても、生徒は教師が見えているものを見られず、結果的に破綻する。

LEADは、この非対称性を3つに分解し、それぞれに対する実用的な解決策を提案する。

### LEADとは結局何なのか

「LEADは手法なのかモデルなのか」という疑問が生じやすいが、答えは「両方を含むシステム全体の名前」。

```
LEAD = 以下3つのセット

  1. LEAD Expert（改良版エキスパートドライバー）
     → State Alignment: エキスパートの「ずるさ」を制限
     → 学習手法・データ生成の改善。モデルの重みではない

  2. TFv6（新しいモデルアーキテクチャ）
     → Intent Alignment: GRU除去、Target Point 3個化
     → 明確なアーキテクチャ変更。重みもHuggingFaceで公開されている

  3. 学習パイプライン
     → 2段階学習（Pre-training → Post-training）
     → マルチデータセット学習
```

アブレーション結果を見ると、各要素の貢献度が分かる:

```
ベースライン (TFv5 + PDM-Lite)        → B2D 83.56
+ State Alignment（手法のみ、同じTFv5） → B2D 84.94  (+1.4)  ← データの質
+ GRU除去（アーキテクチャ変更）          → B2D 87.26  (+2.3)  ← モデル構造
+ Dense Route（アーキテクチャ変更）      → B2D 89.29  (+2.0)  ← モデル構造
+ Full Training（データ量増加）          → B2D 95.0   (+5.7)  ← データ量
```

State Alignmentは「同じモデルでもデータの質で性能が変わる」という手法面の貢献。GRU除去やDense Routeはモデル構造の変更による貢献。そして最後のデータ量増加（40h→73h）が一番大きく効いている。

### TFv5, TFv6 の「TF」とは

TFはTransFuser（Transformer + Fusion）の略。Tübingen大学のAutonomous Visionグループが開発してきたE2E自動運転モデルのシリーズ名。

```
TransFuser v1 (2021, CVPR)   - Prakash et al.
  Transformer Fusionの初提案。単一解像度での融合

TransFuser++ / TFv5 (2023, T-PAMI)   - Chitta et al.
  4段階マルチスケール融合、補助タスク追加、GRU Planning Decoder

TFv6 (2026, CVPR)   - Nguyen et al. (= LEAD)
  GRU除去、Transformer Planning Decoder、Dense Route、Radar Detector
```

v2〜v4は論文として公開されておらず、内部的な改良の版番号と考えられる。公開されているのはv1、v5（TransFuser++）、v6（LEAD）の3つ。

---

## 2. 前提知識: E2E自動運転の基礎

### 勉強トピック: 模倣学習（Imitation Learning）とは？

模倣学習は、エキスパート（人間やルールベースの運転AI）の行動を記録し、ニューラルネットワークに「同じ状況では同じ行動をとれ」と学習させる手法。

```
教師の走行データ: (観測, 行動) のペア
  ↓ 教師あり学習
学習モデル: 観測 → 行動 を出力
```

自動運転における「行動」とは、具体的には以下のもの:
- Waypoints（経路点）: 将来数秒間の車両位置を表す座標列
- Target Speed（目標速度）: 車両が目指すべき速度
- 制御コマンド: ステアリング角度、スロットル、ブレーキ

### 勉強トピック: Open-Loop vs Closed-Loop 評価

| 評価方法 | 説明 | 信頼性 |
|---------|------|--------|
| Open-Loop | 記録済みデータに対して予測精度を計測 | 低い（実走行との乖離が大きい） |
| Closed-Loop | シミュレータ内で実際に車両を走行させて評価 | 高い（実走行に近い） |

Open-Loopでは「次のフレームでどこにいるべきか」の誤差だけ見るが、実際の走行ではその誤差が蓄積（compounding error）して大事故につながる。LEADではClosed-Loop評価を重視する。

### 勉強トピック: Learning by Cheating (LBC) パラダイム

自動運転のE2E学習で広く使われる2段階学習パラダイム:

```
Stage 1: 特権エキスパート（Privileged Expert）
  - シミュレータの内部情報に直接アクセスできる
  - 完璧な判断で走行データを生成

Stage 2: センサベースの学生モデル（Student / Learner）
  - カメラ・LiDAR等のセンサ入力のみ使用
  - Stage 1 で生成されたデータを教師信号として学習
```

従来の研究は Stage 1 のエキスパートを「より賢く・より完璧に」することに注力してきた。LEADは逆に、「エキスパートを生徒に歩み寄らせる」ことの重要性を主張する。

### 勉強トピック: Bird's Eye View (BEV) 表現

自動運転でよく使われる、上空から地面を見下ろした2D表現。

```
        前方 (+X)
          ↑
  左 (+Y) ← → 右 (-Y)
          ↓
        後方 (-X)
```

BEV上に道路構造、他車両、歩行者、信号機などを描画し、ニューラルネットワークの内部表現として使う。典型的な解像度は 4 pixels/meter で、前方64m × 左右40m程度をカバーする。

---

## 3. 核心: Learner–Expert Asymmetry（学習者とエキスパートの非対称性）

LEADが特定した3つの非対称性:

### 3.1 Visibility Asymmetry（視認性の非対称性）

問題: エキスパートはオクルージョン（遮蔽）を無視して全てのアクターを「見える」。学生はカメラの視野内にあるもの、かつ遮蔽されていないものしか知覚できない。

```
エキスパートの視界: 360° 全方位、壁の向こうも見える
学生の視界:         カメラFOV内、遮蔽なしの物体のみ

例: 建物の影から飛び出してくる歩行者
  エキスパート → 事前に減速（見えているから）
  学生         → 飛び出しの瞬間まで分からない

結果: 学生は「なぜここで減速するのか」を理解できないデータで学習してしまう
```

### 3.2 Uncertainty Asymmetry（不確実性の非対称性）

問題: エキスパートはノイズなしの正確な状態情報（速度、加速度、位置）にアクセスできる。学生はセンサのノイズ、検出ミス、推定誤差と戦わなければならない。

```
エキスパートの情報: 前方車両の正確な速度 = 32.5 km/h
学生の推定:        前方車両の推定速度 = 30±5 km/h

エキスパートは「ギリギリ安全」と判断して車間を詰めるが、
学生は同じマージンを維持できない（推定誤差があるため）
```

### 3.3 Intent Asymmetry（意図の非対称性）

問題: 学生に与えられるナビゲーション情報は「Target Point（目標点）1個」だけ。これでは交差点での方向や、ラウンドアバウトでの出口選択などの意図が不十分。

```
ナビゲーション指示の情報量:
  エキスパート: 精密なルート計画（密な経路点列）
  学生:        遠くの目標点1個

問題1: 目標点が遠すぎると経路予測が崩壊
問題2: 目標点が車両の後ろにある場合がある（ラウンドアバウト等）
問題3: 目標点に固執して危険物を無視する（Target Point Bias）
```

---

## 4. 手法1: State Alignment（状態の整合）

エキスパートの「ずるい」部分を制限し、学生が実際に見える情報だけで適切な判断ができるような運転データを生成する。

### 4.1 Visibility Alignment（視認性の整合）

具体的な制約:

| 制約 | 説明 |
|-----|------|
| カメラ視野外のアクターを除外 | エキスパートの衝突チェックから、学生のカメラFOV外にいる動的オブジェクトを除外 |
| 信号機の視認性チェック | カメラFOVに入っていない信号機は無視する |
| 速度上限の推論可能化 | 速度標識を直接読むのではなく、周囲の交通の流れから推論可能な速度上限を使用 |

### 4.2 Uncertainty Alignment（不確実性の整合）

| 制約 | 説明 |
|-----|------|
| 保守的なブレーキング | 衝突コースになくても、観測可能な危険物の近くでブレーキ |
| 低視認性時の減速 | 夜間・大雨時に速度を下げる（学生の知覚精度低下を反映） |
| バウンディングボックスの拡大 | 対向車線からの車両のBBoxを拡大し、保守的なマージン確保 |

### 効果（実験結果）

PDM-Liteデータセット → LEADデータセットに変更した場合（モデルは同じTFv5）:

| ベンチマーク | 変更前 | 変更後 | 改善 |
|-------------|-------|-------|------|
| Longest6 v2 (DS) | 22.51 | 34.05 | +11.5 |
| Bench2Drive (DS) | 83.56 | 84.94 | +1.4 |

ポイント: モデルアーキテクチャを変えずに、教師データの質を改善しただけでこれだけの差が出る。

### 勉強トピック: なぜ「完璧すぎる教師」は悪いのか？

直感に反するが、教師が「上手すぎる」と学生は学べない。

日常の例で考えると分かりやすい: プロ棋士の最善手を真似ようとしても、なぜそれが最善なのかの理由（読み筋）が見えなければ、似たような局面で間違える。自動運転でも、エキスパートが「壁の向こうの歩行者のために減速する」データを見せられても、学生には「理由のない減速」にしか見えず、学習信号がノイズになる。

---

## 5. 手法2: Intent Alignment（意図の整合）

### 5.1 問題: Target Point Bias

従来手法（TFv5）では、ナビゲーション情報として1つの「Target Point」（目標点）をGRUに注入していた。これにより2つの深刻な問題が発生:

問題1: GRUボトルネック

```
TFv5のアーキテクチャ:
  BEVトークン → 6層Transformer Decoder (256次元)
    → GRU (64次元hidden) ← Target Point注入
    → Waypoint出力

問題: 6層256次元のリッチな特徴量を、64次元GRU1層に圧縮している
結果: GRUは最も強い信号（Target Point）に支配され、他の情報が失われる
```

問題2: 速度と操舵の分離

GRUはWaypoint（操舵方向）にのみTarget Pointの情報を注入し、Target Speed（速度）は独立に予測していた。ルートから逸脱するとWaypointとSpeedの整合性が崩れ、暴走や急停止が起きる。

### 5.2 解決策1: GRUの除去とTarget Pointのトークン化

```
TFv6のアーキテクチャ:
  BEVトークン + Target Pointトークン
    → Transformer Decoder (cross-attention)
    → 直接 Waypoint / Route / Speed を出力

変更点:
  - GRUを完全に除去
  - Target Pointを[-1, 1]に正規化してBEVトークンと同列に扱う
  - Waypointも速度もTarget Pointの情報を等しく参照可能
```

効果:

| ベンチマーク | GRUあり | GRUなし | 改善 |
|-------------|--------|--------|------|
| Longest6 v2 (DS) | 34.05 | 40.70 | +6.65 |
| Bench2Drive (DS) | 84.94 | 87.26 | +2.32 |

### 5.3 解決策2: Dense Route Representation（密なルート表現）

Target Pointを1個から3個に増やす:

```
従来: 現在の Target Point のみ
LEAD: [前の Target Point, 現在の Target Point, 次の Target Point]
```

さらに、Target Pointの切り替え閾値距離を短くし、早い段階で次のTarget Pointが見えるようにした。

効果:

| ベンチマーク | TP×1 | TP×3 | 改善 |
|-------------|------|------|------|
| Longest6 v2 (DS) | 40.70 | 42.13 | +1.43 |
| Bench2Drive (DS) | 87.26 | 89.29 | +2.03 |

### 勉強トピック: なぜTarget Point 1個では足りないのか？

交差点を考える。現在のTarget Pointが交差点の反対側、50m先にあるとする。

```
        TP (50m先)
        ↑
  ------+------
        |
        |  ← 自車
```

この時、直進なのか右折なのか左折なのか、Target Point 1個からは判断できない。3個あれば:

```
  TP_prev (通過済み)
        ↓
  ------+------→ TP_next (次の目標)
        |
        ↑
  TP_current (現在の目標)
        |
     自車
```

TP_prev → TP_current → TP_next の軌跡から「右折」が明確になる。

---

## 6. TFv6: TransFuser V6 アーキテクチャ

### 全体構成

```
入力:
  カメラ画像 (3台: 前方左/中央/右) ─→ Image Encoder (RegNetY-032 or ResNet-34)
  LiDAR点群 (BEVラスタ化)          ─→ LiDAR Encoder (RegNet-Y-800MF)
  レーダー (optional)               ─→ Radar Detector (Transformer)

Backbone (TransfuserBackbone):
  Image Features + LiDAR Features
    → 4段階のGPT-style Transformer Fusion
    → 各段階で Image ↔ LiDAR の Cross-Attention
    → 最終的な BEV Feature Map を生成

Decoders (タスク別):
  ├─ PlanningDecoder: Route + Waypoints + Target Speed
  ├─ BEVDecoder: BEVセマンティックセグメンテーション
  ├─ PerspectiveDecoder: カメラ視点のセマンティクス + 深度推定
  └─ CenterNetDecoder: 3Dバウンディングボックス検出
```

### 勉強トピック: TransFuserとは何か？

TransFuser（2021年初出）は、カメラとLiDARの特徴量をTransformerで融合するアーキテクチャ。名前はTransformer + Fusionの造語。

従来手法では特徴量を単純に連結（concatenation）していたが、TransFuserはSelf-Attentionを使って「カメラ画像のどの部分がLiDARのどの部分と対応するか」を自動的に学習する。

```
TransFuser Fusionの仕組み（各レベルで）:

  Image tokens: [I1, I2, ..., In]
  LiDAR tokens: [L1, L2, ..., Lm]
      ↓ 連結
  [I1, ..., In, L1, ..., Lm]
      ↓ Self-Attention（全トークン間で情報交換）
      ↓ FFN（Feed-Forward Network）
      ↓ 分離
  Image tokens' + LiDAR tokens'
```

これを4つの解像度レベル（Feature Pyramid）で行うことで、低解像度（大域的な構造）から高解像度（細部のテクスチャ）まで、マルチスケールな融合が実現する。

### Planning Decoder の詳細

```
入力:
  BEV Feature Map → PlanningContextEncoder → Context Tokens
  レーダー特徴量  → Key/Value として追加

学習可能なクエリ:
  Route Queries (num_route_points個)
  Waypoint Queries (num_waypoints個)
  Speed Query (1個)

Transformer Decoder:
  Queries × Context → Cross-Attention → 各出力ヘッドへ

出力:
  Route:   累積デルタ座標 (正規化済み)
  Waypoint: 将来位置座標 (正規化済み)
  Speed:   離散クラス分類 (回帰ではなくクラス分類)
```

### 勉強トピック: なぜ速度を「回帰」ではなく「分類」で予測するのか？

速度予測を連続値の回帰（例: 32.5 km/h）ではなく、離散的なクラス分類（例: [0-5], [5-10], ..., [55-60] km/h のどのビンか）で行う。

理由:
1. マルチモーダル性: 交差点では「止まる」と「進む」の両方がありうる。回帰は単一の値しか出せないが、分類なら確率分布で表現できる
2. ロバスト性: 外れ値の影響を受けにくい
3. 学習の安定性: クロスエントロピー損失は勾配が安定している

### Radar Detector

レーダーはLiDARほど高密度ではないが、速度情報を直接計測できる点が強み。

```
Radar Detectorのアーキテクチャ:
  生のレーダー検出点 → トークン化（位置 + センサID）
  BEV特徴量 + 自車速度  → コンテキスト
  Transformer Decoder:
    学習可能クエリ × コンテキスト → 検出結果

出力: (x, y, velocity) × N個 + 有効性スコア
マッチング: Hungarian Algorithm で GT と予測を対応付け
```

---

## 7. 学習パイプライン

### 2段階学習

```
Phase 1: Pre-training（事前学習）
  - 全てのPerceptionデコーダ（セマンティクス、深度、BBox、BEV）を有効化
  - PlanningDecoderは無効
  - 目的: Backboneに良い特徴表現を学習させる
  - 期間: 30エポック程度

Phase 2: Post-training（事後学習）
  - PlanningDecoderを有効化
  - Perceptionデコーダはそのまま維持（マルチタスク学習）
  - 事前学習済みモデルの重みをロード
  - 目的: 計画能力を学習させる
  - 期間: 30エポック程度
```

### 勉強トピック: なぜ2段階で学習するのか？

1. Perceptionの基盤が必要: 道路や車両を正しく認識できなければ、良い経路計画はできない
2. 勾配の干渉: 認識タスクと計画タスクを最初から同時に学習すると、勾配が干渉して収束が遅くなる
3. デバッグの容易さ: 段階的に学習することで、問題の切り分けが容易

### データ収集

LEADエキスパート（改良版PDM-Lite）でCARLAシミュレータを走行し、センサデータを記録:

```
記録されるデータ:
  ├─ rgb/                  # 3台のカメラ画像
  ├─ depth/                # 深度マップ
  ├─ semantics/            # セマンティックセグメンテーション
  ├─ lidar/                # LiDAR点群
  ├─ radar/                # レーダー検出
  ├─ bboxes/               # 3Dバウンディングボックス
  ├─ hdmap/                # 自車中心のHDマップ（BEV）
  ├─ metas/                # メタデータ（速度、位置、天候等）
  ├─ rgb_perturbated/      # 摂動を加えたカメラ画像
  ├─ depth_perturbated/    # 摂動を加えた深度
  └─ ...
```

### 勉強トピック: Perturbation（摂動）とは何のためにあるのか？

データ収集時に、カメラリグの位置と角度をランダムにずらしたデータも同時に記録する。

```
通常のカメラ位置:  車両中心
摂動後のカメラ位置: 左右0.1〜1.0m、上下±角度5〜12.5°ずれた位置

目的:
  - 車両がルートから逸脱した状況を疑似的に生成
  - 「ルートに復帰する」行動を学習させる
  - Compounding Errorに対するロバスト性を向上
```

これはDAgger（Dataset Aggregation）の簡易版と考えることができる。実際にルートから外れた走行をしなくても、カメラを動かすことで類似の視覚入力を生成できる。

### マルチデータセット学習

LEADは複数のデータソースで同時に学習できる:

| データソース | 形式 | 用途 |
|------------|------|------|
| CARLA | 3カメラ + LiDAR + Radar | メイン学習 |
| NAVSIM (nuPlan/nuScenes) | 4-8カメラ | 実世界転移 |
| Waymo Open Dataset | 3カメラ | 実世界転移 |

各データソースにはソースラベル（`SourceDataset` enum）が付き、損失関数の重みをデータソース・エポックごとに動的に変更できる（カリキュラム学習）。

### 損失関数

```
Total Loss = Σ (weight_i × loss_i)

Planning損失:
  - Waypoint L1 Loss: 予測経路と教師経路のL1距離
  - Route L1 Loss: 予測ルートと教師ルートのL1距離
  - Speed Cross-Entropy: 速度クラス分類のCE損失

Perception損失:
  - Semantic CE Loss: カメラ視点のセマンティクス
  - Depth L1 Loss: 深度推定
  - BEV Semantic CE Loss: BEVセマンティクス（視錐台マスク付き）
  - CenterNet Loss: ヒートマップ + 回帰（幅/高さ/オフセット/ヨー角）
  - Radar Loss: Hungarian マッチング + L1
```

---

## 8. 実験結果

### CARLAベンチマーク

LEADは主要な3つのCARLAベンチマーク全てでSOTAを達成:

#### 勉強トピック: 各ベンチマークの特徴

| ベンチマーク | ルート数 | 平均距離 | 特徴 |
|-------------|---------|---------|------|
| Bench2Drive | 220ルート | ~150m | 12の町、多様なシナリオ（44種類の交通場面） |
| Longest6 v2 | 36ルート | ~2km | 6つの町の長距離ルート、エラー蓄積が厳しい |
| Town13 | 10ルート | ~12.4km | 学習に含まれない未知の町、汎化性能を測る |

#### 勉強トピック: Driving Score（DS）の計算方法

```
Driving Score = Route Completion × Infraction Score

Route Completion (RC): ルートの何%を走破したか (0〜100)
Infraction Score (IS): 違反のペナルティ (0〜1, 乗算的に減少)

違反の種類:
  - 衝突（歩行者/車両/静的物体）
  - 赤信号無視
  - 一時停止標識無視
  - ルート逸脱
  - タイムアウト

例: RC=80%, 衝突1回(×0.5), 赤信号1回(×0.7)
  → DS = 80 × 0.5 × 0.7 = 28.0
```

#### メイン結果

| 手法 | Bench2Drive DS | Longest6 v2 RC | Town13 DS |
|-----|:-----------:|:------------:|:--------:|
| TFv5 (従来手法) | 83.5 | 23 | 1.08 |
| SimLingo | 85.1 | 22 | - |
| HiP-AD | 86.8 | - | - |
| LEAD TFv6 (ResNet-34) | 94.7 | 57 | 5.01 |
| LEAD TFv6 (RegNetY-032) | 95.2 | 62 | 5.24 |
| LEAD Expert (上限) | 96.8 | 73 | - |

Longest6 v2では従来手法のRC 22-23を62まで改善（約3倍）。

#### アブレーション: 各コンポーネントの貢献

| 変更 | B2D DS | L6v2 DS |
|-----|--------|---------|
| ベースライン (TFv5 + PDM-Lite) | 83.56 | 22.51 |
| + State Alignment | 84.94 | 34.05 |
| + GRU除去 | 87.26 | 40.70 |
| + Dense Route (TP×3) | 89.29 | 42.13 |
| + Full Training (73h) | 95.0 | 54 |

各改善が独立に貢献しており、State AlignmentとIntent Alignmentの両方が重要。

#### センサ構成のアブレーション

| 構成 | B2D DS | L6v2 RC |
|-----|--------|---------|
| Vision Only (カメラのみ) | 91.6 | 43 |
| + LiDAR | 94.7 | 52 |
| + Radar | 94.2 | - |
| + LiDAR + Radar | 95.0 | 54 |

カメラのみでもDS 91.6を達成できるが、LiDARとRadarの追加で更に改善。

#### 推論速度

| モデル | 推論時間/フレーム |
|-------|:---------------:|
| TFv6 ResNet-34 | 40ms (RTX 2080 Ti) |
| TFv6 RegNetY-032 | 70ms (RTX 2080 Ti) |
| SimLingo | >1000ms |

リアルタイム性（<100ms）を十分に満たしている。

---

## 9. 実世界への展開: NAVSIM / Waymo

### 勉強トピック: Sim-to-Real Transfer（シム→実世界転移）

CARLAで学んだモデルを実世界のデータで動かすには「ドメインギャップ」を超える必要がある:
- 画像の見た目（テクスチャ、光条件、物体の多様性）
- 物理挙動（車両ダイナミクス、摩擦係数）
- シーンの複雑さ（交通密度、道路形状）

LEADのアプローチ:
1. CARLAデータは認識タスク（Perception）の事前学習にのみ使用
2. 運転スタイル（Planning）は実世界データで学習（ドメインの運転マナーの違いを吸収）
3. LiDAR/Radarが使えない場合は位置エンコーディングで代替 → LTFv6（Latent TransFuser v6）

### NAVSIM ベンチマーク

| ベンチマーク | LTFv6 (ベースライン) | + LEAD事前学習 | 改善 |
|-------------|:------------------:|:------------:|:----:|
| NAVSIM v1 (PDMS) | 85.4 | 86.4 | +1.0 |
| NAVSIM v2 (EPDMS) | 28.3 | 31.4 | +3.1 |

### Waymo E2E ベンチマーク

| | LTFv6 | + LEAD事前学習 | 改善 |
|--|:-----:|:------------:|:----:|
| RFS | 7.51 | 7.76 | +0.25 |

改善幅は控えめだが、全ベンチマークで一貫して向上。実世界チャレンジでは2位を獲得。

---

## 10. コードベースとの対応

論文の各コンポーネントが、このリポジトリのどこに対応するか:

| 論文のコンセプト | コードの場所 | 主要クラス/ファイル |
|----------------|------------|------------------|
| TFv6モデル全体 | `lead/tfv6/tfv6.py` | `TFv6` |
| TransFuser Backbone | `lead/tfv6/transfuser_backbone.py` | `TransfuserBackbone` |
| GPT Fusion | `lead/tfv6/transfuser_backbone.py` | `GPT` |
| Planning Decoder | `lead/tfv6/planning_decoder.py` | `PlanningDecoder` |
| BEV Decoder | `lead/tfv6/bev_decoder.py` | `BEVDecoder` |
| Perspective Decoder | `lead/tfv6/perspective_decoder.py` | `PerspectiveDecoder` |
| CenterNet Decoder | `lead/tfv6/center_net_decoder.py` | `CenterNetDecoder` |
| Radar Detector | `lead/tfv6/radar_detector.py` | `RadarDetector` |
| LEAD Expert | `lead/expert/expert.py` | `Expert` |
| Privileged Route Planner | `lead/expert/privileged_route_planner.py` | ルート計画とState Alignment |
| 学習ループ | `lead/training/train.py` | `Trainer` |
| 学習設定 | `lead/training/config_training.py` | `TrainingConfig` |
| CARLAデータローダ | `lead/data_loader/carla_dataset.py` | `CARLAData` |
| NAVSIMデータローダ | `lead/data_loader/navsim_dataset.py` | `NavsimData` |
| 閉ループ推論 | `lead/inference/closed_loop_inference.py` | `ClosedLoopInference` |
| センサエージェント | `lead/inference/sensor_agent.py` | `SensorAgent` |
| 閉ループ設定 | `lead/inference/config_closed_loop.py` | 推論時のフラグ設定 |
| 基本設定 | `lead/common/config_base.py` | `BaseConfig` |
| 定数・Enum定義 | `lead/common/constants.py` | `SourceDataset`, `TransfuserBEVSemanticClass` 等 |

### 推論時のフロー

```
SensorAgent.run_step()
  → カメラ/LiDAR/GPS/速度 入力を受信
  → OpenLoopInference.preprocess_input_data() で前処理
  → TFv6.forward() でニューラルネット推論
  → ClosedLoopInference でPID制御器を通して制御コマンド生成
     ├─ Route + Target Speed → PID制御
     └─ Waypoints → PID制御
  → 後処理（停止標識ヒューリスティック、スタック検出）
  → throttle, brake, steer を出力
```

---

## 11. 限界と今後の課題

1. ルート逸脱からの復帰: 学習データはすべてルート上のデータ。大きく逸脱した場合の復帰能力は限定的。DAggerやRL微調整で改善の余地あり
2. 複雑な車線変更: 高速道路の出口で複数回連続して車線変更する場面は苦手
3. Sim-to-Real: 現状はPerceptionの事前学習のみに合成データを使用。Planningまで含めた共同学習は未検証
4. エキスパート設計の汎用性: CARLA固有のルールベースエキスパートに限定した検証。学習型エキスパートや人間のデモンストレーションに原理が適用できるかは未検証

---

## 12. 用語集

| 用語 | 説明 |
|-----|------|
| E2E (End-to-End) | センサ入力から直接制御出力を生成する一気通貫型アプローチ |
| IL (Imitation Learning) | エキスパートの行動を模倣して学習する手法 |
| Closed-Loop | モデルの出力が環境に反映され、次の入力に影響する評価方式 |
| Open-Loop | 記録済みデータに対して予測精度のみを評価する方式 |
| BEV (Bird's Eye View) | 上空からの鳥瞰図表現 |
| DS (Driving Score) | Route Completion × Infraction Score で計算される総合評価指標 |
| RC (Route Completion) | ルートの完走率 (%) |
| IS (Infraction Score) | 違反ペナルティの乗算スコア (0〜1) |
| TransFuser | Transformer-based Sensor Fusion アーキテクチャ |
| LBC (Learning by Cheating) | 特権エキスパート → センサ学生の2段階学習パラダイム |
| Target Point | ナビゲーション用の目標座標点 |
| Waypoint | 将来の車両位置を表す時空間的な経路点 |
| Perturbation | カメラ位置をずらしたデータ拡張手法 |
| GRU | Gated Recurrent Unit（ゲート付き回帰ユニット） |
| PID制御 | Proportional-Integral-Derivative 制御（古典制御手法） |
| Compounding Error | 予測誤差が蓄積して大きくなる現象 |
| DAgger | Dataset Aggregation（ポリシーの失敗状態を含むデータで再学習） |
| LTF (Latent TransFuser) | LiDARの代わりに位置エンコーディングを使うTransFuserのバリアント |
| PDMS | Predictive Driver Model Score（NAVSIM v1の評価指標） |
| EPDMS | Extended PDMS（NAVSIM v2の評価指標） |
| RFS | Rater Feedback Score（Waymo E2Eの評価指標） |
| Hungarian Algorithm | 最小コストマッチング問題を解くアルゴリズム（検出のGT対応付けに使用） |
| PDM-Lite | Planning with Decision Model - Lite（ルールベースエキスパートドライバー） |

---

## 13. 背景技術と関連論文ガイド

LEAD論文が参照している主要な論文を、テーマ別に整理する。LEADの理解に直結するものほど詳しく解説する。

---

### 13.1 E2E自動運転の原点

#### Bojarski et al. [2016] - "End to End Learning for Self-Driving Cars" (NVIDIA)

E2E自動運転の元祖。カメラ画像を入力し、CNNで直接ステアリング角度を出力するという極めてシンプルなアーキテクチャ。NVIDIAの実車で動作を実証した。

```
カメラ画像 → CNN → ステアリング角度
```

この論文以降、「センサから制御まで一気通貫」というE2Eの考え方が自動運転研究の一大テーマになった。ただし、この時点ではLiDARや複数カメラの融合、複雑な都市部での走行は対象外。

#### Codevilla et al. [2018] - "End-to-End Driving via Conditional Imitation Learning" (ICRA)

E2E模倣学習に「条件付き」の概念を導入。ナビゲーションコマンド（直進、左折、右折等）を追加入力として与え、分岐するネットワークヘッドで行動を出力する。

```
カメラ画像 + コマンド（左折/直進/右折）→ CNN → 制御出力
```

LEADのTarget Point方式はこの「条件付き」の発展形。コマンドの代わりに座標点で意図を伝える。

#### Codevilla et al. [2019] - "Exploring the Limitations of Behavior Cloning" (ICCV)

模倣学習（Behavior Cloning）の限界を体系的に分析。主な発見:
- 訓練データ量を増やしても性能が飽和する
- 分布外（out-of-distribution）の状況で急激に性能が劣化する
- Compounding Error が最大の問題

LEADのPerturbation（摂動データ拡張）は、まさにこの分布外問題への対策。

---

### 13.2 Learning by Cheating パラダイム

#### Chen et al. [2019] - "Learning by Cheating" (CoRL)

LEADの学習パラダイムの直接的な祖先。CARLAベンチマークで初の100%成功率を達成。

核心のアイデア:
1. Stage 1: シミュレータの全情報（BEV上の全オブジェクト位置等）にアクセスできる「チート」エージェントを訓練
2. Stage 2: Stage 1 のチートエージェントを教師として、カメラのみのエージェントを蒸留

```
Stage 1: Ground Truth BEV → MLP → 制御 （チート: 完璧に近い性能）
Stage 2: カメラ画像 → CNN → 特徴量 → Stage 1の行動を模倣
```

LEADとの関係: LEADはこのパラダイム自体は維持しつつ、「Stage 1のチートエージェントが上手すぎる（非対称性がある）」という点を問題視し、チートの度合いを制限した。

#### Chen et al. [2021] - "Learning to Drive from a World on Rails" (ICCV)

LBCの発展。世界を「レール上」と仮定し、自車以外は固定的に動くという簡易化で、効率的なオフライン学習を実現。自車の行動のみを学習対象にすることで、状態空間の複雑さを大幅に削減。

#### Chen and Krahenbuhl [2022] - "Learning from All Vehicles" (CVPR)

自車だけでなくシーン内の全車両の行動からも学習する手法。他車両の視点からもデータを生成し、学習データ量を大幅に増加させる。

---

### 13.3 TransFuser 系列（LEADの直接的な技術基盤）

#### Prakash et al. [2021] - "Multi-Modal Fusion Transformer for E2E Autonomous Driving" (CVPR)

TransFuserの初出論文。従来のgeometry-basedなセンサ融合（LiDARの3D点をカメラ画像に射影する等）では、信号機の状態変化のように幾何的には遠いが意味的に重要な情報を統合できないことを指摘。

Transformerの Self-Attention により、カメラとLiDARの特徴量間で大域的な情報交換を実現。衝突を76%削減。

```
カメラ特徴量 ←[Transformer Self-Attention]→ LiDAR特徴量
  ↓                                            ↓
統合された特徴量 → Waypoint予測 → PID制御
```

#### Chitta et al. [2023] - "TransFuser: Imitation with Transformer-based Sensor Fusion" (T-PAMI)

TransFuserの改良版（TransFuser++、TFv5とも呼ばれる）。T-PAMI（トップジャーナル）に掲載。主な改良点:
- Feature Pyramidの4段階での融合（初代は1段階のみ）
- BEVセマンティクス、深度推定、BBox検出などの補助タスク追加
- GRU-based Planning Decoder
- 衝突をさらに48%削減

LEADのTFv6はこのTFv5を直接的に改良したもの。Backbone部分はほぼ同一で、Planning Decoderを置換。

#### 勉強トピック: TransFuserの進化

```
TransFuser v1 (2021, CVPR) - Prakash et al.
  - 単一解像度での Transformer Fusion
  - Waypoint予測のみ
  - CARLAリーダーボード初期、データ品質が低い時代

TransFuser v2 (2021, Master Thesis) - Jaeger
  - アーキテクチャ変更なし
  - 自動ラベリングと補助タスクでデータ品質を改善
  - データだけで大幅に性能向上（アーキテクチャの重要性 vs データの重要性の初期示唆）

TransFuser v3 (2022, T-PAMI) - Chitta et al.
  - ジャーナル版。より良いデータ・センサ・バックボーン・厳密なアブレーション
  - Latent TransFuser（LTF）を初導入: LiDARを位置エンコーディングに置換
  - v1比で約4倍のCARLAリーダーボード性能

TransFuser v3.5 (2024, NeurIPS)
  - v3のアーキテクチャとv4のセンサ設定のハイブリッド

TransFuser v4 / TF++ (2023, ICCV) - Jaeger et al.
  - "Hidden Biases" 論文
  - 改良されたセンサ設定、アーキテクチャ、学習手法
  - CARLAリーダーボード1.0でオープンソース最高性能

TransFuser v5 (2024, Tech Report)
  - CARLAリーダーボード2.0に対応
  - CVPR 2024 CARLA Challenge 2位
  - 4段階マルチスケール Fusion
  - 補助タスク（BEV、深度、セマンティクス、BBox）
  - GRU Planning Decoder、Target Point 1個

TFv6 / LEAD (2026, CVPR) - Nguyen et al.
  - GRU除去、Transformer Planning Decoder
  - Target Point 3個（Dense Route）
  - Radar Detector追加
  - State-Aligned Expert
  - CVPR 2025 Waymo E2E Challenge 2位
```

---

### 13.4 Hidden Biases（隠れたバイアスの発見）

#### Jaeger et al. [2023] - "Hidden Biases of End-to-End Driving Models" (ICCV)

LEADのIntent Alignment手法の直接的な動機となった重要論文。E2Eモデルの性能向上が、アルゴリズムの本質的な改善ではなく、2つの隠れたバイアスに依存していたことを暴露:

1. Target Point Bias（横方向）: モデルが Target Point に向かって積極的にステアリングを切ることで、ルート逸脱から自動復帰する。一見良さそうだが、危険物を無視して目標に突進する副作用がある
2. Longitudinal Averaging Bias（縦方向）: マルチモーダルなWaypoint予測の平均を取ると、暗黙的に減速する傾向がある

この分析を踏まえて開発されたTF++（TransFuser++）は Longest6 で DS +11 を達成。LEADはこの知見をさらに発展させ、GRU除去とDense Routeで Target Point Bias を構造的に解消した。

#### Zimmerlin et al. [2024] - "Hidden Biases of End-to-End Driving Datasets"

上記論文のデータセット版。モデルではなくデータセット側のバイアスを分析:
- エキスパートの運転スタイルが下流ポリシーの性能に大きく影響する
- フレームの「情報量」を推定し、ラベルが変化しないフレームを間引くことでデータセットを縮小しても性能を維持できる

LEADのState Alignment（エキスパートの運転スタイル改善）はこの知見と直接的に関連。

#### Gerstenecker et al. [2025] - "PLANT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving"

Closed-Loop評価自体の構造的欠陥を指摘。評価メトリクスの計算方法やシナリオ設計にバイアスがあり、真の性能を反映していない場合があることを示した。

---

### 13.5 ベンチマークとシミュレータ

#### Dosovitskiy et al. [2017] - "CARLA: An Open Urban Driving Simulator" (CoRL)

LEADの主要評価環境。Unreal Engine 4 ベースのオープンソース自動運転シミュレータ。特徴:
- 複数の都市マップ（Town01〜Town15）
- 動的な天候・照明変化
- 他車両・歩行者・信号機のシミュレーション
- センサ（カメラ、LiDAR、レーダー、GPS等）のシミュレーション
- Python APIによる制御

#### Jia et al. [2024] - "Bench2Drive" (NeurIPS)

CARLAの220ルート、44種類の交通シナリオをカバーする短距離ベンチマーク。各ルートは~150mと短いが、特定のシナリオ（右折時の歩行者横断、信号変化時の対応等）への対処能力を個別に測定できる。

#### Dauner et al. [2024] - "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation" (NeurIPS)

Open-LoopとClosed-Loopの中間に位置する「非反応型シミュレーション」ベンチマーク。実世界データ（nuPlan/nuScenes）を使用。

```
従来のOpen-Loop: 1フレームの予測精度のみ評価
NAVSIM:          数秒間のシミュレーションを展開し、衝突・進捗・快適性を評価
                 ただし他車両は予測に反応しない（非反応型）
従来のClosed-Loop: 他車両も反応する完全なシミュレーション
```

評価指標 PDMS (Predictive Driver Model Score) は以下の要素を統合:
- 衝突回避率
- 進捗（目標に向かって進んでいるか）
- Time-to-Collision（衝突までの時間的余裕）
- 走行可能領域の遵守
- 快適性（急加速・急ハンドルの有無）

#### Xu et al. [2025] - "WOD-E2E: Waymo Open Dataset for E2E Driving"

Waymoの実走行データから安全上重要な稀少シーン（全走行の0.03%未満）を重点的にサンプリングしたベンチマーク。5秒間の軌跡予測を評価。人間アノテーターによる信頼領域内外でのスコアリング（RFS）を採用。

---

### 13.6 エキスパート設計と非対称性の理論

#### Walsman et al. [2023] - "Impossibly Good Experts and How to Follow Them" (ICLR)

LEADの問題意識に最も近い理論的研究。特権情報にアクセスできるエキスパートが、部分観測しかできない学生にとって「不可能なほど良い」行動を示すという問題を形式化。

主張: 特権エキスパートの行動を単純に模倣するのではなく、学生の観測空間で達成可能な最適方策を考慮すべき。

LEADとの違い: Walsmanらは理論的分析が中心。LEADはこの問題を実践的に解決（エキスパートの行動を学生の観測能力に合わせて制約する）。

#### Weihs et al. [2021] - "Bridging the Imitation Gap by Adaptive Insubordination" (ICLR)

エキスパートの指示に「選択的に従わない」ことで、模倣ギャップを埋める手法。学生が自身の観測に基づいて、エキスパートの指示が不適切だと判断した場合に独自の行動を取ることを許容する。

LEADのアプローチとは対照的: Weihsらは学生側を適応させるが、LEADはエキスパート側を適応させる。

#### Messikommer et al. [2025] - "Student-Informed Teacher Training"

教師と学生を共同で訓練し、学生のフィードバックを教師の訓練に反映させるアプローチ。LEADのState Alignmentと思想は近いが、方法論が異なる（LEADはルールベースの制約、こちらは学習ベースの適応）。

---

### 13.7 競合手法・最新手法

#### Renz et al. [2025] - "SimLingo: Vision-Only Closed-Loop Autonomous Driving" (CVPR)

Vision-Language Model（VLM）を活用したE2E運転。言語による状況説明と行動の整合性を学習することで、解釈可能性と性能の両立を目指す。ただし推論に1秒以上かかり、リアルタイム性に課題。

LEADとの比較: LEADのTFv6は40-70msで推論可能。SimLingoはB2D DS 85.1に対し、LEADは95.2。

#### Tang et al. [2025] - "HiP-AD: Hierarchical Planning for Autonomous Driving" (ICCV)

階層的な計画アーキテクチャで、Deformable Attentionを単一デコーダで処理。短距離ベンチマーク（Bench2Drive）に強い。

#### Jia et al. [2025] - "DriveTransformer" (ICLR)

スケーラブルなTransformerアーキテクチャでE2E運転を実現。モデルサイズとデータ量のスケーリング則を検証。

#### Liao et al. [2025] - "DiffusionDrive" (CVPR)

拡散モデルをE2E運転に適用。生成モデルの多様性により、マルチモーダルな将来予測（複数の可能な経路）を生成できる。

#### Li et al. [2024] - "Think2Drive" (ECCV)

強化学習+世界モデルの組み合わせ。模倣学習ではなくRL（報酬設計）で運転ポリシーを学習する代替アプローチ。

#### Jaeger et al. [2025] - "CARL: Learning Scalable Planning Policies with Simple Rewards" (CoRL)

同じく強化学習ベース。シンプルな報酬設計でもスケーラブルな計画ポリシーを学習できることを示した。模倣学習（LEAD）と強化学習（CARL）は同じグループの異なるアプローチ。

---

### 13.8 知覚・表現学習

#### Radosavovic et al. [2020] - "Designing Network Design Spaces" (RegNet)

LEADのバックボーンに使われるRegNetの設計論文。ネットワークの設計空間を体系的に探索し、パラメータ効率の良いアーキテクチャファミリーを発見。

LEADでは:
- Image Encoder: RegNetY-032（メイン）またはResNet-34（軽量版）
- LiDAR Encoder: RegNet-Y-800MF

#### Kerbl et al. [2023] - "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH)

3D Gaussian Splattingの原論文。NAVSIM v2では、Stage 1の予測結果からカメラ視点を仮想的に変化させた画像を3D Gaussian Splattingで生成し、その仮想画像に対してStage 2の評価を行う。

#### 勉強トピック: 3D Gaussian SplattingとNAVSIM v2の関係

```
NAVSIM v2の2段階評価:

Stage 1: 実世界画像でモデルを評価
  → 予測軌跡に基づく自車位置を計算

Stage 2: 予測位置からの視点を3D Gaussian Splattingで生成
  → その仮想画像に対して再度モデルを評価

目的: 自車の予測が少しずれた場合の頑健性を測る
  （自分の予測結果が「次の入力」に影響する、擬似Closed-Loopの実現）
```

---

### 13.9 シミュレーション環境

#### Gulino et al. [2023] - "Waymax" (NeurIPS, Google/Waymo)

大規模マルチエージェントシミュレータ。Waymoの実走行データをベースに、高速なバッチシミュレーションを実現。JAXベースでGPU/TPU上で動作。

#### Kazemkhani et al. [2025] - "GPUDrive" (ICLR)

100万FPSのマルチエージェントシミュレーション。大量の並列シミュレーションにより、強化学習の学習効率を劇的に向上。

#### 勉強トピック: シミュレータの使い分け

| シミュレータ | 特徴 | 用途 |
|------------|------|------|
| CARLA | 高忠実度レンダリング、豊富なセンサモデル | E2Eモデルの閉ループ評価 |
| Waymax | 実データベース、高速バッチ処理 | 大規模RL学習 |
| GPUDrive | 超高速、簡易レンダリング | マルチエージェントRL |
| NAVSIM | 実データ+非反応型シミュレーション | Open-Loop寄りの大規模評価 |

LEADは主にCARLAで閉ループ評価、NAVSIMとWaymo OD E2Eで実世界データ評価を行っている。

---

### 13.10 混合精度学習

#### Micikevicius et al. [2018] - "Mixed Precision Training" (ICLR)

FP16とFP32を混ぜて学習することで、メモリ使用量を削減しつつ精度を維持する手法。LEADの学習パイプラインで使用。

```
通常の学習: 全てFP32 → メモリ大、速度遅い
混合精度:   Forward/BackwardはFP16、重み更新はFP32
  → メモリ約半分、速度約2倍、精度維持

LEADでの使用:
  - torch.cuda.amp.autocast でFP16 Forward
  - GradScaler で勾配スケーリング（FP16の数値範囲制限を回避）
  - スケール値の監視で学習の安定性を確認
```

---

## 14. エキスパートの4段階パイプライン

公式ドキュメントでは、LEAD Expertの制御ロジックは以下の4段階として整理されている:

```
Stage 1: A*グラフ探索で車線トポロジ上の最短経路を計算
  → CARLAのOpenDRIVEマップから車線グラフを構築
  → グローバルルートプランナーで経路点列を生成

Stage 2: 静的障害物を迂回するように経路を修正
  → 工事、駐車車両、事故車両などを検出
  → shift_route_smoothly() でコサイン補間による滑らかな車線変更
  → 緊急車両への譲り、自転車の追い越しにも対応

Stage 3: IDM（Intelligent Driver Model）で基準速度を設定
  → 信号機、停止標識、速度制限に基づく目標速度計算
  → シナリオごとに異なるIDMパラメータ:
    - 停止標識: min_distance=2.0m, time_headway=0.5s
    - 赤信号: min_distance=3.0m, time_headway=0.5s
    - 歩行者: min_distance=4.5m, time_headway=0.125s
    - 先行車: min_distance=4.0m, time_headway=0.25s

Stage 4: 移動障害物との相互作用で速度を微調整
  → 対向車の優先判断
  → 後方からの追突回避（加速して逃げる）
  → 割り込み車両への緊急ブレーキ
  → 横断歩行者の検出と停止
```

### 勉強トピック: IDM（Intelligent Driver Model）とは

車間距離と速度差に基づいて加減速を決定する古典的な車両追従モデル。1999年にTreiber et al.が提案。パラメータは少ないが、人間の運転行動をよく近似する。

```
加速度 a = a_max * [1 - (v/v0)^4 - (s*/s)^2]

  v:   現在速度
  v0:  目標速度
  s:   先行車との車間距離
  s*:  望ましい車間距離 = s0 + v*T + v*Δv/(2*sqrt(a_max*b))
  s0:  最小車間距離 (min_distance)
  T:   希望車頭時間 (time_headway)
  b:   快適減速度
```

LEADでは、シナリオの種類（歩行者、信号、先行車等）ごとにmin_distanceとtime_headwayを変えることで、状況に応じた適切な車間距離を実現している。

---

## 15. Known Issues と実践Tips

### 既知の問題（公式ドキュメントより）

1. マルチGPU学習での性能微減
   - 4GPU学習がシングルGPUより閉ループ性能がわずかに低い場合がある
   - 差は僅かだが一貫して発生
   - 最大性能を求める場合はシングルGPU学習を検討

2. PID制御器の限界
   - デフォルトのPID制御器は最適ではない
   - MPC（Model Predictive Control）に置き換えるとBench2Drive DSが5-7点改善する予備実験結果あり
   - LEADの性能はさらに伸びる余地がある

3. CARLA 0.9.16の非互換性
   - goal-pointパイプラインに問題があり、ポリシーの動作が劣化する
   - CARLA 0.9.15を使うこと

4. PyTorch static_graphの回避
   - static_graphパラメータを使うと性能が低下するため、意図的に回避している

### 評価結果のばらつき

CARLAには内在的な確率性があり、同じチェックポイントでも結果がブレる:

```
典型的なばらつき:
  Bench2Drive:  ±1-2 DS
  Longest6 v2:  ±5-7 DS
  Town13:       ±1.0 DS
```

推奨される評価プロトコル:
- 最低: 3モデル（異なるシード）× 1評価 = 3回の平均
- 推奨: 3モデル × 3評価 = 9回の平均

### CARLAの運用Tips

- 起動の約10%が失敗する（ポート競合、GPU初期化エラー）
- 複数ルートを同じCARLAインスタンスで評価するとレンダリングバグが出る → ルート間でCARLAを再起動する
- Longest6 v2とTown13のルートをBench2Driveの評価リポジトリで評価しないこと（メトリクス定義が異なる）

```bash
# ルート間のCARLA再起動パターン
bash scripts/clean_carla.sh
bash scripts/start_carla.sh
```

### 評価時間の目安（16-32 GTX 1080 Ti使用時）

| ベンチマーク | 3シード | 備考 |
|------------|:------:|------|
| Bench2Drive (220ルート) | ~1日 | 短距離ルートで高速 |
| Longest6 v2 (36ルート) | ~1日 | 長距離だがルート数少 |
| Town13 (10ルート) | ~2日 | 1ルート12kmと非常に長い |

動画生成を無効化すると大幅に高速化できる。

---

## 参考文献

```bibtex
@inproceedings{Nguyen2026CVPR,
  author = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
  title = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026},
}
```
