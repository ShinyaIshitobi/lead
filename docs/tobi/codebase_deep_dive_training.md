# lead/training/ ディレクトリ詳解

このドキュメントでは、LEADプロジェクトの訓練パイプラインを構成する `lead/training/` ディレクトリの各ファイルについて、E2E(End-to-End)自動運転の初学者向けに詳しく解説する。

---

## 目次

1. [全体像](#全体像)
2. [train.py - 訓練ループ](#trainpy---訓練ループ)
3. [config_training.py - 設定パラメータ](#config_trainingpy---設定パラメータ)
4. [logger.py - ロギング](#loggerpy---ロギング)
5. [mixed_training_utils.py - マルチデータセット訓練](#mixed_training_utilspy---マルチデータセット訓練)
6. [rfs.py - Rater Feedback Score (Waymo)](#rfspy---rater-feedback-score-waymo)
7. [training_utils.py - 初期化ヘルパー](#training_utilspy---初期化ヘルパー)

---

## 全体像

`lead/training/` は、LEADモデル(TransFuser v6ベース)の訓練に関する全てのコードを格納している。訓練パイプラインの流れは以下のとおり。

```
設定読み込み → キャッシュ初期化 → PyTorch分散処理初期化 → モデル構築 → データローダ構築 → オプティマイザ構築
→ エポックループ開始
    → データシャッフル → バッチサイズ更新 → 損失重みスケジュール → 訓練 → チェックポイント保存
→ エポックループ終了
```

[学習トピック] E2E自動運転とは、カメラやLiDARなどのセンサ入力から、走行経路(ウェイポイント)やステアリング/速度指令を直接予測するアプローチのことを指す。従来のモジュラー方式(認識 → 計画 → 制御)と異なり、一つのニューラルネットワークで入力から出力までを一気通貫で学習する。

---

## train.py - 訓練ループ

ファイルパス: `lead/training/train.py`

### Trainerクラス

訓練の中核となるクラス。`__init__`、`train_loop`、`train`、`save` の4つの主要メソッドで構成される。

### __init__ - 初期化

初期化は以下の順序で実行される。

1. `initialize_config()` - 設定ファイルの読み込み
2. `initialize_training_session_cache()` - ディスクキャッシュの準備(データロード高速化用)
3. `initialize_torch()` - 乱数シード設定、分散処理初期化、CUDA設定
4. `initialize_model()` - TFv6モデルの構築、チェックポイント復元、DDP(DistributedDataParallel)ラップ
5. `initialize_dataloader()` - データセットとサンプラーの構築
6. `initialize_optimizer()` - AdamW + CosineAnnealing + GradScalerの構築

マルチGPU環境で `DistributedDataParallel` が適用される場合、`self.model` には内部のモジュール(`.module`)を参照する。これにより、損失計算やチェックポイント保存時にDDPのラッパーを意識せずにモデルにアクセスできる。

### train_loop - エポックループ

各エポックで以下の処理が順番に実行される。

1. `dataloader.dataset.shuffle(epoch)` - データセットのシャッフル
2. `sampler.update_batch_sizes(epoch)` - バッチサイズ比率の更新(マルチデータセット訓練時にエポックごとにデータセット間の比率を変える)
3. `schedule_loss_weights(epoch)` - 損失関数の重み更新
4. `train()` - 1エポック分の訓練
5. `save(rfm_score)` - モデルチェックポイント保存(rank 0のみ)
6. `torch.distributed.barrier()` - 全プロセスの同期

`schedule_loss_weights` は、各データセット(CARLA、NavSim、Waymo E2E)に対して `config.detailed_loss_weights()` を呼び出し、返された全ての重みを合計が1になるよう正規化する。

### train - バッチ単位の訓練

1バッチあたりの処理は以下のとおり。

1. `torch.amp.autocast` コンテキスト内でフォワードパス実行(混合精度)
2. モデル出力から損失を計算(`model.compute_loss`)
3. 各損失項に対応する重みを掛けて加重和を取る
4. autocastコンテキストの外側でバックワードパス実行(`scaler.scale(loss).backward()`)
5. `scaler.step(optimizer)` + `scaler.update()` で勾配ステップ
6. ロガーによるメトリクス記録
7. `optimizer.zero_grad(set_to_none=True)` で勾配をクリア

[学習トピック] 混合精度訓練(Mixed Precision Training)とは、float32とfloat16(またはbfloat16)を組み合わせて訓練することで、GPU メモリ使用量を削減しつつ訓練速度を向上させる手法である。`autocast` がフォワードパス中に自動的に精度を切り替え、`GradScaler` がfloat16使用時の勾配アンダーフローを防ぐ。

### 勾配オーバーフロー検出

`scaler.step()` の前後で `scaler.get_scale()` の値を比較し、スケールが減少した場合はオーバーフローが発生したと判断する。この場合、そのステップでの勾配更新はスキップされ、`gradient_steps_skipped` カウンタがインクリメントされる。また、スケーラーのスケール値が `grad_scaler_max_grad_scale`(デフォルト: 2^16)を超えないよう制限される。

オーバーフローが検出されなかった場合のみ、学習率スケジューラが更新される。

### RFMスコアによるベストモデル選択

Waymo E2Eデータを使用している場合、各エポック終了時に `evaluate_waymo_e2e()` を実行してRFM(Rater Feedback Metric)スコアを算出する。このスコアに基づいて、最良モデルが `model_best_rfm_{epoch}_{score}.pth` として保存される。過去のベストモデルより良いスコアが出た場合、古いベストモデルは削除される。

### save - チェックポイント保存

保存対象は以下の5つ。

- モデルの state_dict
- オプティマイザの state_dict
- スケジューラの state_dict
- GradScalerの state_dict
- gradient_steps_skippedの値(テキストファイル)

ストレージ節約のため、前エポックのチェックポイントは削除されるが、`epoch_checkpoints_keep` で指定されたエポック(2のべき乗の累積和: 7, 15, 31, 63, ...)は保持される。これにより、訓練の途中経過を一定間隔で振り返ることが可能になる。

ZeroRedundancyOptimizer使用時は、保存前に `optimizer.consolidate_state_dict(0)` を呼び出して全GPUのオプティマイザ状態をGPU 0に集約する。

---

## config_training.py - 設定パラメータ

ファイルパス: `lead/training/config_training.py`

約1120行、100以上のパラメータを持つ設定クラス `TrainingConfig`。`BaseConfig` を継承し、環境変数 `LEAD_TRAINING_CONFIG` やJSONファイルから設定を読み込む。`@overridable_property` デコレータが付いたプロパティは外部から上書き可能。

### ターゲットデータセットとカメラ

- `target_dataset` - 訓練対象のデータセット。CARLAデータのルートパスや使用データセットの組み合わせから自動判定される
- `num_available_cameras` - データセットに応じたカメラ台数(CARLA: 3台、NavSim: 4台、Waymo: 3台)
- `used_cameras` - 各カメラの有効/無効リスト。上書き可能

### 計画領域(Planning Area)

BEV(Bird's Eye View)座標系での計画範囲を定義する。

- `pixels_per_meter` = 4.0 - BEVグリッドにおける1メートルあたりのピクセル数
- `min_x_meter` / `max_x_meter` - 前後方向の範囲。データセットにより -32〜32m または 0〜64m
- `min_y_meter` / `max_y_meter` - 左右方向の範囲。-32〜32m または -40〜40m

[学習トピック] BEV(Bird's Eye View)とは、上空から俯瞰した視点の表現で、自動運転では2D平面上にセンサ情報を投影して周囲環境を表す。pixels_per_meter = 4.0 の場合、1メートルが4ピクセルで表現される。計画領域が -32〜32m の場合、64m x 64m の範囲を 256 x 256 ピクセルのグリッドで表すことになる。

### 訓練パラメータ

- `epochs` - 訓練エポック数。CARLA: 31、NavSim: 61、Waymo E2E: 20
- `batch_size` - ローカル環境: 2、Slurm環境: 64
- `lr` = 3e-4 - 基本学習率
- `weight_decay` = 0.01 - L2正則化の強さ
- `seed` = 0 - 再現性のための乱数シード

### 混合精度設定

- `torch_float_type` - A100/L40S GPUでは bfloat16、その他では float32
- `use_mixed_precision_training` - A100/L40S GPUで自動的に有効化
- `need_grad_scaler` - float16使用時のみ必要(bfloat16ではダイナミックレンジが広いため不要)

[学習トピック] bfloat16はfloat16と同じ16ビットだが、指数部が8ビット(float32と同じ)で仮数部が7ビットの浮動小数点形式。float16より表現可能な数値範囲が広いため、勾配スケーリングなしでも訓練が安定しやすい。A100以降のGPUでハードウェアサポートがある。

### GradScaler設定

- `grad_scaler_init_scale` = 1024 - 初期スケール値
- `grad_scaler_growth_factor` = 2.0 - スケール拡大係数
- `grad_scaler_backoff_factor` = 0.5 - オーバーフロー時のスケール縮小係数
- `grad_scaler_growth_interval` = 256 - スケール拡大までのステップ間隔
- `grad_scaler_max_grad_scale` = 2^16 - スケールの上限値

### オプティマイザとスケジューラ

- オプティマイザ: AdamW (`amsgrad=True`, `fused=True`)
- オプション: `ZeroRedundancyOptimizer` (分散訓練でメモリ節約)
- スケジューラ: `CosineAnnealingWarmRestarts` (T_0=steps_per_epoch, T_mult=2)。Waymo E2Eのみ `CosineAnnealingLR` (リスタートなし)

[学習トピック] CosineAnnealingWarmRestartsは、学習率をコサインカーブに沿って減衰させ、一定ステップ後にリスタート(学習率を元に戻す)するスケジューラ。T_mult=2 は、リスタートのたびに周期が2倍になることを意味する。例えばT_0=1000の場合、最初の周期は1000ステップ、次は2000ステップ、その次は4000ステップとなる。

### モデル設定

- `image_architecture` = "resnet34" - 画像エンコーダのバックボーン
- `lidar_architecture` = "resnet34" - LiDARエンコーダのバックボーン
- `LTF` = False - Latent TransFuserモード
- `freeze_backbone` = False - バックボーンの重み固定

### プランニングデコーダ

- `transfuser_token_dim` = 64 - トランスフォーマのトークン次元数
- `transfuser_num_bev_cross_attention_layers` = 6 - BEVクロスアテンション層数
- `transfuser_num_bev_cross_attention_heads` = 8 - アテンションヘッド数
- `use_planning_decoder` = False - プランニングデコーダの使用有無(事前訓練ではFalse)

### 速度予測

- `target_speed_classes` = [0, 4, 8, 10, 13.89, 16, 17.78, 20] m/s - 離散速度クラス

これらの値はそれぞれ 0 km/h、約14.4 km/h、約28.8 km/h、36 km/h、約50 km/h、約57.6 km/h、約64 km/h、72 km/h に対応する。

### ウェイポイント予測

- `num_way_points_prediction` - CARLA: 8 (4Hz x 2秒)、NavSim: 8 (2Hz x 4秒)、Waymo: 10 (2Hz x 5秒)
- `waypoints_spacing` - CARLA: 5 (4Hz)、NavSim/Waymo: 10 (2Hz)

[学習トピック] ウェイポイント(waypoint)とは、車両が将来辿るべき経路上の点列のことである。例えば2Hz x 5秒なら、0.5秒間隔で5秒先までの10個の(x, y)座標点を予測する。モデルはこれらの将来位置を教師あり学習で予測し、推論時にはコントローラがこの点列に沿って車両を制御する。

### レーダー設定

- `num_radar_queries` = 20 - レーダートランスフォーマのクエリ数
- `radar_token_dim` = 256 - レーダートークン次元数
- `radar_num_layers` = 4 - レーダートランスフォーマ層数
- `radar_num_heads` = 8 - レーダーアテンションヘッド数
- `radar_num_points_per_sensor` = 75 - センサあたりのレーダーポイント数

レーダー入力はCARLA Leaderboardモードでのみ有効。

### バウンディングボックス検出

- `detect_boxes` = True - BBox補助タスクの使用
- `max_num_bbs` = 90 - 検出可能な最大バウンディングボックス数
- `num_dir_bins` = 12 - オブジェクト向きの離散化ビン数
- `bb_confidence_threshold` = 0.3 - 検出受け入れの信頼度閾値

CenterNetベースの検出ヘッドが使用され、ヒートマップ、幅/高さ、オフセット、ヨー角、速度の各ロスが計算される。

### マルチデータセット設定

- `use_carla_data` = True - CARLAデータの使用
- `use_navsim_data` = False - NavSimデータの使用
- `use_waymo_e2e_data` = False - Waymo E2Eデータの使用
- `mixed_data_training` - 複数データセットの同時使用を自動検出

### 損失重みの動的スケジューリング

`detailed_loss_weights(source_dataset, epoch)` メソッドは、データセットとエポックに応じて各損失項の重みを返す。

主な損失項と規則:

- セマンティックセグメンテーション、デプス、レーダー、速度推定のロスはCARLAデータのみで有効
- BEVセマンティック、BBox関連のロスは全データセットで有効
- プランニング関連ロス(waypoints, target_speed, spatial_route)はプランニングデコーダ有効時のみ
- 非CARLAデータセットの損失キーにはプレフィックス(例: `navsim_`, `waymo_e2e_`)が付与される

---

## logger.py - ロギング

ファイルパス: `lead/training/logger.py`

### Loggerクラス

TensorBoardとWandB(Weights & Biases)への二重ロギングを担当する。ロギングは rank 0 のプロセスのみが実行する。

### 初期化

- TensorBoard: `SummaryWriter` で `logdir` にログを書き込む
- WandB: プロジェクト名は `lead_pretrain`(事前訓練時)または `lead_posttrain`(事後訓練時)
- WandBの設定(config全体)、実験名(description)、レジューム設定が反映される

### スカラーロギング (log_train)

一定頻度(`log_scalars_frequency`)で以下のメトリクスを記録する。

デバッグメトリクス:
- エポック番号、学習率、バッチサイズ、GPU数
- モデルパラメータ数、データセットサイズ
- 訓練進捗率、残りステップ数
- GPU最大メモリ使用量(GB)
- データロード時間(平均)
- GradScalerのスケール値
- 勾配ステップスキップ回数

損失メトリクス:
- `unscaled_loss/` プレフィックス: 重みなしの生の損失値
- `scaled_loss/` プレフィックス: 重み付き損失値
- `metric/` プレフィックス: モデル固有のカスタムメトリクス

### 画像ロギング

一定頻度(`log_images_frequency`)で `visualize_sample()` を呼び出し、訓練サンプルの予測結果を可視化する。ローカル環境ではPNG保存、Slurm環境ではWandBにアップロードされる。CARLAモードでのみ有効。

---

## mixed_training_utils.py - マルチデータセット訓練

ファイルパス: `lead/training/mixed_training_utils.py`

複数の異種データセット(CARLA、NavSim、Waymo E2E)を同時に使って訓練するための仕組みを提供する。

[学習トピック] 自動運転の訓練では、シミュレータ(CARLA)で大量のデータを安価に生成できるが、現実世界(Waymo等)のデータとの分布の違い(domain gap)が課題となる。複数データセットを混ぜて訓練することで、シミュレーションの豊富なデータ量と現実世界のリアリズムを両立させるSim-to-Real転移を試みる。

### MixedDataset

複数データセットをインターリーブ方式でラップするクラス。

インデックスの計算方式:
```
index = dataset_idx + dataset_sample_idx * num_datasets
```

例えば2つのデータセット(ds0, ds1)がある場合:
- index 0 → ds0のサンプル0
- index 1 → ds1のサンプル0
- index 2 → ds0のサンプル1
- index 3 → ds1のサンプル1

全データセットのサイズが同一であることが前提条件。

### MixedSampler

`BatchSampler` を継承し、各データセットからの取得比率をスケジューラに基づいて制御する。

- `update_batch_sizes(epoch)` でエポックごとにバッチ内のデータセット比率を更新
- `__iter__` では各データセットのサンプラーからスケジュール通りの個数を取得し、MixedDatasetのグローバルインデックスに変換してバッチを生成
- バッチサイズの合計 x GPU数 が `config.batch_size` と一致することを保証

### UniformSampleScheduler

全データセットから均等にサンプリングする最もシンプルなスケジューラ。各データセットの比率は `1/データセット数` で固定される。

### Sim2RealSampleScheduler

CARLAデータと実世界データの比率をエポックに応じて徐々にシフトさせるスケジューラ。

アンカーエポックとバッチ配分(総バッチサイズ64の場合):

| エポック | CARLA | リアルデータ |
|---------|-------|-------------|
| 0       | 40    | 24          |
| 1       | 32    | 32          |
| 3       | 24    | 40          |
| 7       | 16    | 48          |
| 15      | 8     | 56          |
| 31      | 0     | 64          |

訓練初期はシミュレーションデータで基礎的な特徴表現を学習し、後半では実世界データの比率を上げてドメイン適応を図る戦略である。

### mixed_data_collate_fn

異種データセットをバッチにまとめる際のカスタムcollate関数。

一部のデータセットにしか存在しないキー(例: CARLAのみのセマンティックセグメンテーションラベル)を、欠損サンプルではゼロ埋めして統一的なバッチを構成する。テンソルにはゼロテンソル、NumPy配列にはゼロ配列、文字列には空文字列がデフォルト値として使われる。

---

## rfs.py - Rater Feedback Score (Waymo)

ファイルパス: `lead/training/rfs.py`

Waymo Open Dataset E2E Driving Challenge 2025 の公式評価メトリクス「Rater Feedback Score (RFS)」の実装。

[学習トピック] RFSは人間のレーター(評価者)が作成した「望ましい走行軌跡」に対して、モデルの予測軌跡がどれだけ近いかを測る指標である。従来の単純な軌跡距離(ADE/FDE)よりも、横方向・縦方向の安全性を区別して評価でき、人間の運転品質評価により近い指標とされる。

### アルゴリズム概要

1. レーター指定軌跡の前処理(パディング/切り詰め)
2. 変位ベクトルの計算による縦方向/横方向の分解
3. 推論軌跡とレーター軌跡の間の縦方向/横方向距離の算出
4. 3秒時点と5秒時点での距離をフィルタリング
5. 信頼領域(trust region)の判定
6. 信頼領域外のスコア減衰(指数減衰)
7. 推論軌跡の確率で重み付けした最終スコアの算出

### 信頼領域(Trust Region)

推論軌跡がレーター軌跡から一定の閾値内にある場合、「信頼領域内」と判定され、レーターが付けたスコアがそのまま適用される。

基本閾値:
- 3秒時点: 1.0m
- 5秒時点: 1.8m

速度によるスケーリング:
```
scale = clip(0.5 + 0.5 * (init_speed - 1.4) / (11 - 1.4), 0.5, 1.0)
```

低速(1.4 m/s以下)では閾値が半分に、高速(11 m/s以上)では閾値がそのまま適用される。速度が低いほど精密な経路追従が求められるという直感に一致する。

### 減衰スコアリング

信頼領域外では、距離に応じた指数減衰が適用される。

```
exponent = max(normalized_distance - 1.0, 0.0)
decay = 0.1 ^ exponent
```

信頼領域の境界(normalized_distance = 1.0)ではスコアに変化なし。境界の2倍の距離では元のスコアの10%になる。

信頼領域外のスコアの最小値は 4.0 に設定されている(スコアは0〜10の範囲)。

### 横方向/縦方向の方向ベクトル

レーター軌跡の各タイムステップでの変位ベクトルから縦方向(進行方向)を求め、90度回転で横方向を算出する。車両が停止している場合(変位ゼロ)は前のタイムステップの方向を引き継ぐ。最初のタイムステップで停止している場合は(1, 0)(車両座標系のx軸方向)がデフォルトとなる。

### compute_rfs 関数

上位レイヤーから呼ばれるラッパー関数。デフォルトで4Hz、5秒の軌跡を評価する。推論確率は均一(全て1.0)で、単一軌跡の評価に特化している。

---

## training_utils.py - 初期化ヘルパー

ファイルパス: `lead/training/training_utils.py`

Trainerクラスの初期化処理を個別の関数に分離したモジュール。

### initialize_config

1. `TrainingConfig()` でデフォルト設定をロード
2. `config.load_file` が指定されている場合、そのチェックポイントと同じディレクトリの `config.json` を読み込んで設定を復元

### initialize_training_session_cache

`diskcache.Cache` を使用してディスクキャッシュを初期化する。サイズ上限は 2TB (2048 GB)。SSD上に配置され、データロード時のI/Oボトルネックを軽減する。Slurm環境では `/scratch/$SLURM_JOB_ID`、それ以外では `$SCRATCH` 環境変数または `/tmp/$USER` に配置される。

### initialize_torch

以下の設定を行う。

乱数シード:
- `torch.manual_seed`、`np.random.seed`、`random.seed`、`torch.cuda.manual_seed`、`torch.cuda.manual_seed_all` を全て同じシードで初期化

分散処理:
- GPU 2台以上の場合、NCCLバックエンドで `torch.distributed.init_process_group` を実行
- タイムアウトは120分

CUDA最適化:
- `tf32` 演算の有効化(行列積の高速化)
- `cudnn.benchmark = True`(入力サイズに応じた最適カーネル選択)
- `cudnn.deterministic = False`(再現性より速度を優先)

ワーカー数は `(CPUコア数 / GPU数) * workers_per_cpu_cores` で計算される。

### initialize_model

1. `TFv6` モデルインスタンスを作成しCUDAに配置
2. (オプション) `SyncBatchNorm` への変換
3. 全ての正規化層をfp32にパッチ(`fn.patch_norm_fp32`) - 数値安定性のため
4. チェックポイントからの復元(`load_file` 指定時)
5. バックボーンの凍結設定
6. `channels_last` メモリフォーマットへの変換(畳み込み層の高速化)
7. マルチGPU環境での `DistributedDataParallel` ラップ
8. `torch.compile` によるJITコンパイル(fullgraph, max-autotune, inductorバックエンド)

[学習トピック] `torch.compile` はPyTorch 2.0で導入されたJITコンパイル機能。モデルの計算グラフを解析し、カーネルフュージョンやメモリ最適化を自動適用する。`max-autotune` モードではCUDAグラフも活用され、カーネル起動のオーバーヘッドが最小化される。

### initialize_optimizer

1. AdamWオプティマイザの構築(amsgrad=True, fused=True)
2. ZeroRedundancyOptimizer使用時は、各GPUがパラメータの一部のみのオプティマイザ状態を保持(メモリ節約)
3. 学習率スケジューラの構築
4. GradScalerの構築(float16使用時のみ有効)
5. チェックポイントからの状態復元(continue_failed_training時)

[学習トピック] ZeroRedundancyOptimizerはDeepSpeed ZeROに着想を得た最適化手法。通常のデータ並列では全GPUがオプティマイザの完全なコピーを持つが、ZeROではパラメータを分割して各GPUが一部だけ管理する。これにより、オプティマイザのメモリ使用量が GPU数分の1 に削減される。

### initialize_dataloader

1. 使用するデータセットの構築(CARLA、NavSim、Waymo E2E)
2. 各データセットに対する `DistributedSampler` の作成(shuffle=True, drop_last=True)
3. サンプルスケジューラの選択(`Sim2RealSampleScheduler` または `UniformSampleScheduler`)
4. `MixedDataset` による複数データセットの統合
5. `MixedSampler` によるバッチ生成戦略の設定
6. `DataLoader` の構築(pin_memory, prefetch_factor=16, persistent_workers=True)

### seed_worker

DataLoaderの各ワーカープロセスに固有のシードを割り当てる関数。

```
worker_seed = torch.initial_seed() % 2^32 + rank * 1000
```

PyTorchの `initial_seed` はワーカーごとに異なるが、GPU(rank)間では同一となる。そのため、rankを1000倍して加算することで、異なるGPUの同じワーカー番号でも異なるシードが得られるようにしている。

### set_start_method

DataLoaderのマルチプロセス生成方式を設定する。

- `fork` - 親プロセスのメモリ空間をコピー(Linux推奨、コード編集中でも動作可能)
- `spawn` - 新しいPythonプロセスを起動(全OS対応、安全だが遅い)
- `forkserver` - フォールバック

---

## 各ファイルの依存関係

```
train.py
  ├── training_utils.py (初期化処理)
  ├── logger.py (ロギング)
  └── config_training.py (設定)

training_utils.py
  ├── config_training.py (設定)
  ├── mixed_training_utils.py (マルチデータセット)
  └── (各データセットクラス: CARLAData, NavsimData, WODE2EData)

mixed_training_utils.py
  └── config_training.py (設定)

rfs.py
  └── (NumPy, 外部依存なし)

logger.py
  ├── config_training.py (設定)
  └── (TensorBoard, WandB)
```

---

## まとめ

lead/training/ ディレクトリは、E2E自動運転モデルの訓練に必要な以下の機能を提供している。

- train.py: 訓練メインループ。混合精度、勾配スケーリング、分散訓練、チェックポイント管理を統合
- config_training.py: 100以上の訓練パラメータを一元管理。データセット/GPU/訓練フェーズに応じた動的な設定切り替え
- logger.py: TensorBoardとWandBへの二重ロギング。スカラー/画像メトリクスの定期記録
- mixed_training_utils.py: 異種データセット(シミュレーション+実世界)の混合訓練基盤。Sim-to-Real転移のためのスケジューリング
- rfs.py: Waymo E2E Challenge 2025の公式評価指標。信頼領域ベースの軌跡品質評価
- training_utils.py: PyTorch分散訓練、モデルコンパイル、キャッシュなどの初期化処理を集約
