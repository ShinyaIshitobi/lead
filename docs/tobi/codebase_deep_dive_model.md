# lead/tfv6/ -- モデルアーキテクチャ詳解

このドキュメントでは、LEADの中核となるニューラルネットワークモデルが実装されている `lead/tfv6/` ディレクトリを詳しく解説する。対象読者はEnd-to-End自動運転の初学者を想定しており、各コンポーネントの役割と仕組みを基礎から説明する。

---

## 目次

1. [全体像と設計思想](#全体像と設計思想)
2. [tfv6.py -- メインモデル](#tfv6py----メインモデル)
3. [transfuser_backbone.py -- マルチモーダル特徴融合](#transfuser_backbonepy----マルチモーダル特徴融合)
4. [planning_decoder.py -- 経路計画デコーダ](#planning_decoderpy----経路計画デコーダ)
5. [bev_decoder.py -- BEVセマンティックセグメンテーション](#bev_decoderpy----bevセマンティックセグメンテーション)
6. [perspective_decoder.py -- 画像デコーダ](#perspective_decoderpy----画像デコーダ)
7. [center_net_decoder.py -- 3D物体検出](#center_net_decoderpy----3d物体検出)
8. [radar_detector.py -- レーダー検出](#radar_detectorpy----レーダー検出)
9. [tfv5_planning_decoder.py -- レガシー経路計画デコーダ](#tfv5_planning_decoderpy----レガシー経路計画デコーダ)
10. [transfuser_utils.py -- ユーティリティ関数群](#transfuser_utilspy----ユーティリティ関数群)
11. [学習のためのキーワード集](#学習のためのキーワード集)

---

## 全体像と設計思想

LEADのモデルは、カメラ画像とLiDAR(またはその代替表現)を入力として受け取り、自動運転に必要な複数のタスクを同時に解く「マルチタスクモデル」である。具体的には以下のタスクを扱う:

- 経路計画(Planning): 車両が将来どこを走るべきかのウェイポイント予測
- BEVセマンティックセグメンテーション: 鳥瞰図(Bird's Eye View)での道路・車両等のセグメンテーション
- 画像セマンティックセグメンテーション: カメラ画像でのピクセル単位のクラス分類
- 深度推定(Depth Estimation): カメラ画像からの距離推定
- 3D物体検出(Bounding Box Detection): BEV上での車両・歩行者等の検出
- レーダー検出(Radar Detection): レーダーセンサーデータを用いた物体検出

このモデルはCARLAシミュレータとNavSIMデータセットの両方に対応しており、データソースに応じて一部のデコーダが別インスタンスとして生成される。

```
学習トピック: End-to-End自動運転
  従来の自動運転では「認識 → 予測 → 計画」を別々のモジュールで処理していたが、
  End-to-Endアプローチではセンサー入力から直接行動(ウェイポイントや速度)を出力する。
  LEADはこのEnd-to-Endアプローチを採用しつつ、認識タスクを補助タスクとして
  同時に学習させることで、内部表現の質を高めている。
```

---

## tfv6.py -- メインモデル

ファイルパス: `lead/tfv6/tfv6.py`

### TFv6クラス

`TFv6(nn.Module)` はモデル全体を統括するトップレベルのクラスである。設定(config)に応じてバックボーンと各デコーダを条件付きでインスタンス化する。

### コンポーネントの条件付き生成

TFv6のコンストラクタでは、configのフラグに応じて必要なデコーダのみが生成される:

| 条件 | 生成されるデコーダ |
|---|---|
| `use_semantic and use_carla_data` | `PerspectiveDecoder`(セマンティック用） |
| `use_depth and use_carla_data` | `PerspectiveDecoder`（深度用） |
| `use_bev_semantic and use_carla_data` | `BEVDecoder`（CARLA用） |
| `use_bev_semantic and use_navsim_data` | `BEVDecoder`（NavSim用） |
| `detect_boxes and use_carla_data` | `CenterNetDecoder`（CARLA用） |
| `detect_boxes and use_navsim_data` | `CenterNetDecoder`（NavSim用） |
| `radar_detection and use_carla_data` | `RadarDetector` |
| `use_planning_decoder` | `PlanningDecoder` または `TFv5PlanningDecoder` |

CARLAとNavSimで別インスタンスが作られるのは、クラス数やラベル体系が異なるためである。

### forwardパイプライン

forwardメソッドのデータフローは以下の順序で実行される:

```
入力データ(data)
    |
    v
[1] backbone(data) --> bev_features, image_features
    |
    v
[2] radar_detector(bev_features, data) --> radar_features, radar_predictions  (任意)
    |
    v
[3] planning_decoder(bev_features, radar, data) --> route, waypoints, speed  (任意)
    |
    v
[4] semantic_decoder(data, image_features) --> pred_semantic  (任意)
    |
    v
[5] depth_decoder(data, image_features) --> pred_depth  (任意)
    |
    v
[6] backbone.top_down(bev_features) --> bev_feature_grid
    |
    v
[7] center_net_decoder(data, bev_feature_grid) --> pred_bounding_box  (任意)
    |
    v
[8] bev_semantic_decoder(bev_feature_grid) --> pred_bev_semantic  (任意)
    |
    v
Prediction dataclass にまとめて返却
```

### Prediction dataclass

`Prediction` はモデルの全出力を型付きで保持するデータクラスである。jaxtyping によるテンソル形状のアノテーションが付いており、各フィールドは `None` を許容する（対応するデコーダが無効の場合）。

主なフィールド:
- `pred_future_waypoints`: 将来のウェイポイント座標 (bs, n_waypoints, 2)
- `pred_target_speed_distribution`: 目標速度の分布 (bs, num_speed_classes)
- `pred_target_speed_scalar`: 目標速度のスカラー値 (bs,)
- `pred_route`: ルート予測 (bs, n_checkpoints, 2)
- `pred_bev_semantic`: BEVセマンティック予測 (bs, classes, H, W)
- `pred_bounding_box`: CenterNetによるバウンディングボックス予測

### compute_loss メソッド

各デコーダの `compute_loss` を順次呼び出し、損失を辞書に集約して返す。これにより、学習ループ側ではモデル内部の構造を意識せずに損失の合計を計算できる。

---

## transfuser_backbone.py -- マルチモーダル特徴融合

ファイルパス: `lead/tfv6/transfuser_backbone.py`

```
学習トピック: マルチモーダル融合(Multi-Modal Fusion)
  自動運転ではカメラ、LiDAR、レーダーなど複数のセンサーを使う。
  これらの異なるモダリティ(データ種類)の情報を統合的に扱うことを
  「マルチモーダル融合」と呼ぶ。TransFuserは画像とLiDARの特徴を
  Transformerで融合するアーキテクチャである。
```

### TransfuserBackbone クラス

デュアルストリーム(2分岐)エンコーダ構造を持ち、画像ブランチとLiDARブランチの特徴を4段階で融合する。

#### 画像エンコーダ
- TIMMライブラリの事前学習済みモデル（ResNet/RegNetなど）を使用
- `config.image_architecture` でアーキテクチャを指定
- `pretrained=True` でImageNet事前学習の重みをロード
- `features_only=True` で中間特徴量を抽出するモードで動作

#### LiDARエンコーダ
- 同じくTIMMモデルを使用するが、事前学習なし（`pretrained=False`）
- 入力チャネル数はLTFモードの場合は2、そうでなければ1
- LiDARの疑似画像（Pseudo-image）を処理する

#### LTFモード(Latent TransFuser)
LTFモードが有効な場合、実際のLiDAR入力の代わりに位置エンコーディンググリッドが使われる。これは`[0, 1]`の範囲で正規化されたx座標とy座標の2チャネルのグリッドであり、LiDARセンサーが存在しない環境でもモデルが動作できるようにするための手法である。

```python
lidar[:, 0] = y_grid  # 縦方向の位置エンコーディング
lidar[:, 1] = x_grid  # 横方向の位置エンコーディング
```

### GPTクラス -- Transformer融合モジュール

各解像度レベルでの融合を担当する。GPTという名前だが、因果的なマスクは使わず、双方向のSelf-Attentionを行う。

処理の流れ:
1. 画像トークンとLiDARトークンをチャネル方向にflattenしてシーケンスとして並べる
2. 学習可能な位置埋め込み(pos_emb)を加算
3. Transformerブロック（Block）を通す
4. 出力をイメージトークンとLiDARトークンに分割し、元の空間形状に戻す

### fuse_features メソッド

各解像度レベルで以下の手順を実行する:

1. AdaptiveAvgPool2dでアンカーポイント数にプーリング
2. 1x1 Convでチャネル数を合わせる
3. GPT(Transformer)で融合
4. 元の解像度にバイリニア補間で戻す
5. 残差接続(Residual Connection)で元の特徴に加算

```
学習トピック: 残差接続(Residual Connection)
  特徴量に変換結果を「加算」することで、勾配が直接流れるパスを確保する。
  これにより深いネットワークでも安定した学習が可能になる。
  ResNetで提案された手法であり、現代の多くのアーキテクチャで使われている。
```

### top_down メソッド

FPN(Feature Pyramid Network)に類似した構造で、BEV特徴を段階的にアップサンプリングする:

```
bev_features (高チャネル, 低解像度)
    |
    v
c5_conv (1x1 Conv) --> チャネル数をbev_features_chanelsに変換
    |
    v
upsample --> up_conv5 (3x3 Conv + ReLU)
    |
    v
upsample2 --> up_conv4 (3x3 Conv + ReLU)
    |
    v
bev_feature_grid (低チャネル, 高解像度)
```

### Block / SelfAttention クラス

標準的なTransformerブロック。Pre-Normalization方式（先にLayerNormを適用してからAttention/FFNを通す）を採用している:

```
x --> LayerNorm --> SelfAttention --> + (残差) --> LayerNorm --> MLP --> + (残差) --> 出力
      ^                              |             ^                     |
      |______________________________|             |_____________________|
```

SelfAttentionではPyTorchの `scaled_dot_product_attention` を使用しており、FlashAttentionなどの最適化が自動的に適用される。活性化関数はGELUではなくReLUを使用している点が特徴的である。

### 主要パラメータ

| パラメータ | 説明 |
|---|---|
| `img_vert_anchors` / `img_horz_anchors` | 画像特徴のプーリング先サイズ |
| `lidar_vert_anchors` / `lidar_horz_anchors` | LiDAR特徴のプーリング先サイズ |
| `n_head` | Attentionのヘッド数 |
| `n_layer` | Transformerブロックの層数 |
| `block_exp` | FFNの拡張率 |

---

## planning_decoder.py -- 経路計画デコーダ

ファイルパス: `lead/tfv6/planning_decoder.py`

```
学習トピック: 自動運転における経路計画
  経路計画は「車両がどのような経路を走るか」を決定するタスク。
  LEADでは以下の3つを同時に予測する:
  - ルート(route): 大まかな走行経路のチェックポイント列
  - ウェイポイント(waypoints): 時間情報を含む詳細な走行点列
  - 目標速度(target speed): 次に出すべき速度
```

### PlanningDecoder クラス

Transformerデコーダベースの経路計画モジュール。BEV特徴と車両のステータス情報をコンテキストとして、学習可能なクエリからルート/ウェイポイント/速度を予測する。

#### アーキテクチャ

```
BEV特徴 + ステータス情報
    |
    v
PlanningContextEncoder --> コンテキストトークン列
    |                              |
    |                              v (Key/Value)
    |                      TransformerDecoder
    |                              ^
    |                              | (Query)
    v                     学習可能クエリ(route_queries + waypoint_queries + speed_query)
    |
    v
route_decoder / wp_decoder / target_speed_decoder
    |
    v
ルート / ウェイポイント / 速度分布
```

#### 学習可能クエリ

クエリの数はconfigの設定によって動的に決まる:
- `predict_spatial_path`: ルート予測用クエリ（num_route_points_prediction個）
- `predict_temporal_spatial_waypoints`: ウェイポイント予測用クエリ（num_way_points_prediction個）
- `predict_target_speed`: 速度予測用クエリ（1個）

これらのクエリは連結されて一つの `nn.Parameter` として管理される。

#### cumsumによる経路の連続性

ルートとウェイポイントの出力には `torch.cumsum` が適用される。これにより、デコーダは「各点の絶対座標」ではなく「前の点からの差分」を学習する。cumsumを使うことで経路の連続性が自然に保たれ、飛び飛びの予測を防ぐ効果がある。

```python
route = torch.cumsum(self.route_decoder(route_queries), 1)
waypoints = torch.cumsum(self.wp_decoder(waypoints_queries), 1)
```

### PlanningContextEncoder クラス

BEV特徴と車両のステータス情報をトークン列としてエンコードする。以下の情報をトークンに変換してコンテキストに追加できる:

| 情報 | トークン数 | 説明 |
|---|---|---|
| 速度(velocity) | 1 | 現在の車速をmax_speedで正規化 |
| 加速度(acceleration) | 1 | 現在の加速度 |
| コマンド(command) | 1 | ナビゲーションコマンド(直進/左折/右折など) |
| ターゲットポイント(target_point) | 1 | 次に向かうべき目標地点 |
| 前後のターゲットポイント | 各1 | 前後のナビゲーション目標 |
| 過去の位置(past_positions) | N | 過去N時刻分の自車位置 |
| 過去の速度(past_speeds) | N | 過去N時刻分の自車速度 |
| レーダー(radar) | Q | レーダー検出結果のトークン |

BEV特徴はコサイン位置埋め込み(PositionEmbeddingSine)が加算された後、flattenされてトークン列となる。ステータストークンにはそれぞれ学習可能な位置埋め込みが加算される。

### Two-Hot エンコーディング

速度予測には「Two-Hot エンコーディング」という手法が使われている。

```
学習トピック: Two-Hot エンコーディング
  連続値（例: 速度 6.5 m/s）をクラス分類問題として扱うための手法。
  離散的な速度クラス [0, 4, 8, 12, ...] が定義されており、
  実際の速度値を隣接する2つのクラスに線形補間で割り当てる。

  例: 速度 6.0 m/s、クラス = [0, 4, 8]
    --> 4と8の間に位置するので、4に重み0.5、8に重み0.5のラベルを生成
    --> [0.0, 0.5, 0.5]

  ブレーキ時は常にクラス0(速度0)に1.0を割り当てる。
  デコード時は各クラスの速度値との加重平均で元のスカラー値に戻す。
```

### 損失関数

| 予測対象 | 損失関数 |
|---|---|
| ウェイポイント | L1損失(Mean Absolute Error) |
| ルート | L1損失(ADE) + L1損失(FDE: 最終点のみ) |
| 目標速度 | Cross-Entropy損失(Two-Hot分布に対して) |
| ヘディング(NavSim) | L1損失 |

ADE(Average Displacement Error)は全ウェイポイントの平均誤差、FDE(Final Displacement Error)は最終地点の誤差を表す。

---

## bev_decoder.py -- BEVセマンティックセグメンテーション

ファイルパス: `lead/tfv6/bev_decoder.py`

```
学習トピック: BEV(Bird's Eye View)表現
  BEVは「鳥瞰図」、つまり真上から見下ろした視点でのマップ表現。
  自動運転では車両周囲の環境を2Dマップとして表現するのに使われる。
  各ピクセルが実空間の一定面積(例: 0.25m x 0.25m)に対応し、
  道路、車両、歩行者などのクラスでラベル付けされる。
```

### BEVDecoder クラス

シンプルな畳み込みベースのデコーダで、BEV特徴グリッドからセマンティックセグメンテーションマップを生成する。

ネットワーク構造:
```
BEV特徴グリッド (B, bev_features_channels, H, W)
    |
    v
3x3 Conv + ReLU
    |
    v
1x1 Conv (チャネル数をクラス数に変換)
    |
    v
Bilinear Upsample (元のLiDARグリッド解像度に拡大)
    |
    v
BEVセマンティック予測 (B, num_classes, lidar_H, lidar_W)
```

### フラスタム可視マスク

BEVグリッドの全領域がカメラから見えるわけではない。カメラの視野角の外にある領域は予測の信頼性が低いため、学習時に除外する。`valid_bev_pixels` マスクはカメラのフラスタム(錐台)内にある領域のみを1.0として保持し、それ以外を-1に設定する。`F.cross_entropy` の `ignore_index=-1` によって、不可視領域のピクセルは損失計算から除外される。

### CARLA/NavSim別インスタンス

CARLAとNavSimではBEVセマンティックのクラス数が異なるため、別々のBEVDecoderインスタンスが生成される。`source_dataset` フラグにより、バッチ内の各サンプルが適切なデコーダで処理されるよう損失がマスクされる。

---

## perspective_decoder.py -- 画像デコーダ

ファイルパス: `lead/tfv6/perspective_decoder.py`

### PerspectiveDecoder クラス

バックボーンの画像特徴から、元の画像解像度に近いセグメンテーションマップまたは深度マップを復元するデコーダ。3段階のプログレッシブアップサンプリングを行う。

```
image_features (B, in_channels, H_low, W_low)
    |
    v
deconv1: 3x3 Conv --> ReLU --> 3x3 Conv --> ReLU
    |
    v
Bilinear Interpolate (scale_factor_0 倍)
    |
    v
deconv2: 3x3 Conv --> ReLU --> 3x3 Conv --> ReLU
    |
    v
Bilinear Interpolate (scale_factor_1 倍)
    |
    v
deconv3: 3x3 Conv --> ReLU --> 3x3 Conv
    |
    v
(必要に応じて最終解像度調整)
    |
    v
出力 (B, out_channels, final_H, final_W)
```

### セマンティック vs 深度

同じクラスで両方のモダリティに対応:
- `modality="semantic"`: out_channels = num_semantic_classes、損失はCross-Entropy
- `modality="depth"`: out_channels = 1、損失はL1、出力時にsqueeze(1)でチャネル次元を除去

---

## center_net_decoder.py -- 3D物体検出

ファイルパス: `lead/tfv6/center_net_decoder.py`

```
学習トピック: CenterNet
  CenterNetはアンカーフリーの物体検出手法。各物体を「中心点」として表現し、
  ヒートマップから中心点を検出する。NMS(Non-Maximum Suppression)の代わりに
  局所最大値フィルタとTop-K選択を使うため、高速に動作する。
  LEADではBEV空間上でCenterNetを適用し、車両や歩行者を検出する。
```

### CenterNetDecoder クラス

BEV特徴グリッドから複数のブランチ(ヘッド)で物体の属性を予測する。

| ブランチ | 出力チャネル | 説明 |
|---|---|---|
| heatmap_head | num_classes | クラスごとの中心点ヒートマップ |
| wh_head | 2 | バウンディングボックスの幅と高さ |
| offset_head | 2 | 中心点の小数点以下のオフセット |
| yaw_class_head | num_dir_bins | 向きの離散クラス |
| yaw_res_head | 1 | 向きの残差(クラス内の微調整値) |
| velocity_head | 1 | 速度(複数時刻のLiDARを使う場合のみ) |

各ヘッドは `3x3 Conv --> ReLU --> 1x1 Conv` のシンプルな構造である。

### 損失関数

ヒートマップにはGaussian Focal Lossを使用する。これは通常のFocal Lossをガウシアンターゲット向けに修正したものである:

```
pos_loss = -log(pred) * (1 - pred)^alpha * 正例マスク
neg_loss = -log(1 - pred) * pred^alpha * (1 - target)^gamma
```

alpha=2, gamma=4 がデフォルトで、正例(物体の中心)付近では予測ミスを大きくペナルティし、負例(背景)付近ではガウシアンカーネルに近い領域のペナルティを軽減する。

その他の損失:
- wh, offset: L1損失(pixel_weightでマスク)
- yaw_class: Cross-Entropy損失
- yaw_res: Smooth L1損失
- velocity: L1損失

### Top-K キーポイント抽出

推論時には以下の手順で物体を検出する:

1. ヒートマップにMax Poolingベースの局所最大値フィルタを適用
2. 全クラス全ピクセルからTop-K個のスコア最大点を選択
3. 各キーポイントからwh, offset, yaw, velocityを収集
4. 画像座標系から車両座標系に変換

### CenterNetBoundingBoxPrediction / PredictedBoundingBox

`CenterNetBoundingBoxPrediction` はモデル出力の生データを保持するdataclass。`cached_property` を使い、画像座標系・車両座標系でのバウンディングボックス変換を遅延評価する。

`PredictedBoundingBox` はポストプロセス済みの個別バウンディングボックスを表すfrozenなdataclass。座標変換やスケーリングのユーティリティメソッドを持つ。

### 角度のクラス+残差エンコーディング

物体の向き(yaw)は離散クラスと残差の組み合わせで表現される:
```
連続角度 = クラスの中心角度 + 残差
```
例えばnum_dir_bins=12なら、各クラスは30度の範囲をカバーし、残差で微調整する。デコード時は `class2angle` 関数で連続角度に変換する。

---

## radar_detector.py -- レーダー検出

ファイルパス: `lead/tfv6/radar_detector.py`

```
学習トピック: レーダーセンサーと物体検出
  レーダーはカメラやLiDARと異なり、悪天候でも安定して動作する。
  直接的に物体の相対速度を計測できるのが大きな利点。
  ただし空間解像度が低いため、他のセンサーとの融合が重要になる。
```

### RadarDetector クラス

レーダーの生データをBEV特徴と融合し、トランスフォーマーデコーダで物体検出を行うモジュール。

#### 入力のトークン化

レーダーの各点(最大300点)を以下の情報からトークンに変換する:

```
レーダー点の位置(x, y)
    --> BEV特徴マップからバイリニアサンプリング
    --> サンプリングした特徴量 (D次元)

+ 相対速度 (1次元、max_speedで正規化)
+ センサーID (4次元、one-hotエンコーディング)
    |
    v
radar_point_tokenizer (MLP)
    |
    v
+ サイン位置埋め込み (BEV座標を[0,1]に正規化してから生成)
    |
    v
レーダートークン (B, 300, D)
```

#### Transformerデコーダ

学習可能なクエリ(num_radar_queries個)がKey/Valueとしてのコンテキストトークン(BEVトークン + 自車速度トークン + レーダートークン)に対してCross-Attentionを行う。

出力は各クエリに対して:
- state_decoder: (x, y, velocity)の3次元を出力、tanhで範囲を制限
- label_decoder: 有効/無効の1次元ロジットを出力

#### ハンガリアンマッチング

```
学習トピック: ハンガリアンマッチング(Hungarian Matching)
  予測されたN個のクエリとN個のGT(Ground Truth)を1対1で対応付ける
  最適割り当て問題を解くアルゴリズム。DETRで導入されたこの手法により、
  NMSを使わずに物体検出の学習が可能になる。
  scipyのlinear_sum_assignmentで実装されている。
```

コスト行列は以下の2つのコストの重み付き和で構成される:
- 状態コスト: 予測とGTの(x, y, v)のL1距離(スケール正規化あり)
- 分類コスト: 有効/無効のBinary Cross-Entropy

マッチング後、対応するペアに対してL1損失(状態)とBCE損失(分類)を計算する。

---

## tfv5_planning_decoder.py -- レガシー経路計画デコーダ

ファイルパス: `lead/tfv6/tfv5_planning_decoder.py`

### TFv5PlanningDecoder クラス

LEADの前身であるTransFuser v5で使われていたGRU(Gated Recurrent Unit)ベースの経路計画デコーダ。現在は `config.use_tfv5_planning_decoder` フラグで有効化できる。

```
学習トピック: GRU (Gated Recurrent Unit)
  GRUはLSTMの簡略版のリカレントニューラルネットワーク。
  時系列データの処理に使われ、前の時刻の情報を隠れ状態として伝播する。
  ウェイポイントの逐次予測(前の点を考慮して次の点を予測)に適している。
```

v6のPlanningDecoderとの主な違い:
- Transformerデコーダの出力をGRUに通して順序依存性を持たせている
- ターゲットポイントの座標をGRUの初期隠れ状態としてエンコード
- ルート用とウェイポイント用に別々のGRUを持つ
- レーダー情報やNavSIMへの対応がない(より単純な構造)

PlanningContextEncoderも簡略版で、速度とコマンドの2つのステータストークンのみに対応する。

---

## transfuser_utils.py -- ユーティリティ関数群

ファイルパス: `lead/tfv6/transfuser_utils.py`

このファイルにはモデル全体で共通して使われるユーティリティ関数が集約されている。

### normalize_imagenet

入力画像をImageNetの統計量(平均・標準偏差)で正規化する。TIMMの事前学習モデルはImageNet正規化された入力を前提としているため、この変換が必要。

```
R: (pixel/255 - 0.485) / 0.229
G: (pixel/255 - 0.456) / 0.224
B: (pixel/255 - 0.406) / 0.225
```

### patch_norm_fp32

モデル内の全ての正規化レイヤー(BatchNorm, GroupNorm, LayerNorm)の演算をFP32で強制する。Mixed Precision学習時に、正規化レイヤーをFP16で計算すると数値不安定性が生じるため、この対策が必要。forwardメソッドをモンキーパッチして、入力をFP32に変換 → 正規化 → 元のdtypeに戻す処理を行う。

### force_fp32

デコレータとして使用し、指定した関数の引数を自動的にFP32に変換してから実行する。`apply_to` パラメータで変換対象の引数名を指定できる。CenterNetのGaussian Focal Lossなど、数値精度が重要な計算で使用される。

### gen_sineembed_for_position

2D座標を正弦波ベースの位置埋め込みに変換する。DAB-DETRから採用した手法で、[0, 1]に正規化された(x, y)座標を高次元の位置ベクトルに変換する。

```
学習トピック: 正弦波位置埋め込み(Sinusoidal Positional Embedding)
  Transformerは入力の順序情報を持たないため、位置埋め込みで位置を伝える。
  正弦波位置埋め込みは異なる周波数のsin/cosの組み合わせで位置を表現する。
  学習可能な位置埋め込みに比べ、未知の位置にも汎化しやすいという利点がある。
```

### unit_normalize_bev_points

BEV座標(メートル単位)を[0, 1]の範囲に正規化する。正弦波位置埋め込みの入力として使用される。

```
x_normalized = (x - min_x) / (max_x - min_x)
y_normalized = (y - min_y) / (max_y - min_y)
```

### bev_grid_sample

BEV特徴マップ上の任意の座標から双線形補間(bilinear interpolation)で特徴量をサンプリングする。PyTorchの`F.grid_sample`を使用し、メートル単位の絶対座標を[-1, 1]のグリッド座標に変換してからサンプリングする。レーダー検出でレーダー点の位置からBEV特徴を取得する際に使用される。

### class2angle

離散角度クラスと残差から連続角度値を復元する。CenterNetのyaw予測のデコードに使用される。

```
angle = class_index * (2*pi / num_dir_bins) + residual
```

結果は[-pi, pi]の範囲に制限される。

---

## 学習のためのキーワード集

End-to-End自動運転の学習を進めるにあたり、このコードベースに関連する重要なキーワードをまとめる。

| キーワード | 説明 |
|---|---|
| TransFuser | カメラとLiDARの特徴をTransformerで融合するアーキテクチャ。本コードベースの基盤。 |
| BEV (Bird's Eye View) | 鳥瞰図表現。上空から見下ろした2Dマップとして環境を表現する。 |
| TIMM | PyTorch Image Models。画像認識用の事前学習済みモデルのライブラリ。 |
| CenterNet | アンカーフリーの物体検出手法。中心点ヒートマップで物体を表現する。 |
| FPN (Feature Pyramid Network) | マルチスケールの特徴マップを統合する構造。top_downメソッドが該当。 |
| Hungarian Matching | 予測とGTの最適1対1マッチング。DETR系の物体検出で使用。 |
| Two-Hot Encoding | 連続値を隣接する2つの離散クラスに分配するエンコーディング手法。 |
| Mixed Precision Training | FP16とFP32を混合して学習を高速化する手法。force_fp32やpatch_norm_fp32が関連。 |
| Positional Embedding | Transformerに位置情報を付与する手法。学習可能なものと固定(正弦波)のものがある。 |
| Residual Connection | 入力をショートカットして加算する接続。勾配消失問題を緩和する。 |
| GRU | Gated Recurrent Unit。時系列処理用のリカレントネットワーク。v5デコーダで使用。 |
| ADE / FDE | Average/Final Displacement Error。経路予測の評価指標。 |
| Focal Loss | クラス不均衡に対応した損失関数。CenterNetのヒートマップ学習で使用。 |
| CARLA | オープンソースの自動運転シミュレータ。学習データの主要なソース。 |
| NavSIM | nuPlanベースの自動運転ベンチマーク。実世界データでの評価に使用。 |
| jaxtyping | テンソルの形状をアノテーションとして記述し、実行時に検証するライブラリ。 |
| beartype | Pythonの型ヒントを実行時に検証するライブラリ。LEADでは広く使用されている。 |
