# コードベース詳細解説: スクリプト・ツール・ノートブック編

このドキュメントでは、LEADプロジェクトの `scripts/`、`slurm/`、`notebooks/` およびその他のツール群について解説する。
E2E（End-to-End）自動運転の初学者を対象とし、各スクリプトの役割・使い方・背景知識を網羅的に説明する。

---

## 学習トピック一覧

このドキュメントを読むことで、以下のトピックを学ぶことができる。

- CARLAシミュレータの環境構築と起動・停止の方法
- 分散学習（DDP: Distributed Data Parallel）の仕組みと実行方法
- SLURMクラスタでのジョブ管理と実験ワークフロー
- ベンチマーク評価（Bench2Drive、Longest6、Town13）の実行方法
- エキスパートデータ収集のパイプライン
- データキャッシュとバケット構築によるI/O最適化
- Jupyter Notebookを使ったデータ分析と可視化
- Flaskベースの違反分析Webダッシュボード
- プロジェクトの環境設定とパス管理

---

## 1. lead/leaderboard_wrapper.py - 評価エントリポイント

### 概要

`leaderboard_wrapper.py` は、エキスパートエージェントとモデルエージェントの両方を統一的に評価するためのラッパースクリプトである。
CARLAのLeaderboard評価をサブプロセスとして起動し、評価モードに応じて適切なリーダーボード・エージェント・設定を自動的に選択する。

### 学習トピック: E2E自動運転における評価とは

E2E自動運転では、モデルの性能を測定するために「Leaderboard評価」と呼ばれる仕組みを用いる。
CARLAシミュレータ上で事前に定義されたルートを走行し、交通ルールの遵守、衝突の有無、目的地への到達率などを総合的に評価する。
LEADではこの評価を複数のリーダーボード形式（STANDARD、BENCH2DRIVE、AUTOPILOT）でサポートしている。

### 主要コンポーネント

LeaderboardType（列挙型）:

- STANDARD: 標準的なCARLA Leaderboard 2.0 評価
- BENCH2DRIVE: Bench2Driveベンチマーク用の評価（短いルートが多い）
- AUTOPILOT: エキスパートデータ収集用の評価

ModeConfig クラス:

- `get_mode_config()` メソッドがCLI引数を解析し、使用するリーダーボードの種類、エージェントファイル、設定ファイル、チェックポイント、トラックを決定する
- エキスパートモード（`--expert`）の場合は `lead/expert/expert.py` を使用
- Py123Dモード（`--py123d`）の場合は `lead/expert/expert_py123d.py` を使用
- モデル評価の場合は `lead/inference/sensor_agent.py` を使用

### CLI引数

```
--checkpoint   : モデルのチェックポイントディレクトリ
--routes       : 評価に使用するルートXMLファイル
--expert       : エキスパートモードで実行
--bench2drive  : Bench2Driveリーダーボードを使用
--py123d       : Py123Dエキスパートを使用
--port         : CARLAサーバーのポート番号
```

### 新しい評価モードの追加方法

1. `ModeConfig.get_mode_config()` に新しいパラメータと条件分岐を追加
2. `main()` 関数に対応するCLI引数を追加
3. パイプラインの残りの部分は自動的に新モードを処理する

---

## 2. scripts/ - クイックスタートスクリプト集

`scripts/` ディレクトリには、ローカル環境での開発・テスト用スクリプトが格納されている。
CARLAの環境構築から、学習・評価・データ収集まで、一通りの操作を素早く実行できる。

### 2.1 CARLA環境管理

#### setup_carla.sh - CARLAのダウンロードとインストール

```
scripts/setup_carla.sh
```

CARLA 0.9.15 のLinuxバイナリと追加マップをダウンロードし、`3rd_party/CARLA_0915/` に展開する。

処理内容:
1. `3rd_party/CARLA_0915` ディレクトリを作成
2. CARLA 0.9.15 の本体をダウンロード・展開
3. 追加マップ（AdditionalMaps_0.9.15）をダウンロード・展開
4. `ImportAssets.sh` を実行してアセットをインポート

学習トピック: CARLAシミュレータとは

CARLAはオープンソースの自動運転シミュレータで、Unreal Engine上に構築されている。
LEADでは CARLA 0.9.15 を使用しており、複数の都市マップ（Town01〜Town15）が利用可能。
追加マップにはTown06、Town07、Town10、Town11、Town12、Town13、Town15などが含まれる。

#### start_carla.sh - CARLAの起動

```
scripts/start_carla.sh [ポート番号]
```

CARLAサーバーをバックグラウンドで起動する。デフォルトのポートは2000。

起動オプション:
- `-quality-level=Poor`: 低品質レンダリング（GPU負荷を軽減）
- `-world-port=2000`: ワールドポート
- `-resx=800 -resy=600`: レンダリング解像度
- `-nosound`: サウンド無効化
- `-graphicsadapter=0`: GPU 0を使用
- `-RenderOffScreen`: ヘッドレスモード（ディスプレイなしで実行可能）

ポート番号を引数として渡すことで、複数のCARLAインスタンスを異なるポートで同時実行できる。

#### clean_carla.sh - CARLAプロセスの強制終了

```
scripts/clean_carla.sh
```

GPU 0 上で動作している CarlaUE4 プロセスを `nvidia-smi` で検出し、`kill -9` で強制終了する。
CARLAがハングした場合やプロセスが残留した場合に使用する。

#### reset_carla_world.py - ワールドのリセット（再起動なし）

```
python scripts/reset_carla_world.py [--host HOST] [--port PORT] [--reload-map]
```

CARLAサーバーを再起動せずに、ワールド内のアクター（車両、歩行者、センサー）を全て破棄する。
信号機はグリーンにリセットされ、物理状態も10ティックかけてクリアされる。

残留アクターが検出された場合は2回目のクリーンアップパスが自動実行される。
`--reload-map` オプションを指定すると、マップ自体を再読み込みして完全なクリーンアップを行う。

学習トピック: CARLAのアクターとワールド

CARLAの世界は「アクター」の集合で構成される。車両（vehicle）、歩行者（walker）、センサー（sensor）、
信号機（traffic_light）などが代表的なアクター。評価やデータ収集を繰り返す際、前回の実行で残ったアクターが
干渉することがあるため、このようなクリーンアップスクリプトが必要になる。

### 2.2 学習スクリプト

#### pretrain_ddp.sh - 分散事前学習

```
scripts/pretrain_ddp.sh
```

torchrunを使用した分散データ並列（DDP）学習スクリプト。
利用可能なGPU数を `nvidia-smi` から自動検出し、マルチGPU学習を実行する。

設定される環境変数:
- `OMP_NUM_THREADS`: CPUコア数に設定
- `OPENBLAS_NUM_THREADS=1`: NumPyのマルチスレッドを無効化（スレッドの二重生成を防止）
- `NCCL_P2P_DISABLE=1`: NCCL P2P通信を無効化（一部環境での互換性対策）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: CUDAメモリアロケータの最適化

出力先: `outputs/local_training/pretrain`

学習トピック: 分散データ並列（DDP）学習

DDPは複数のGPUで同じモデルを並列に学習する手法。各GPUが異なるデータバッチを処理し、勾配を同期する。
`torchrun` はPyTorchの分散学習ランチャーで、プロセスの起動・通信バックエンドの設定を自動化する。
`--standalone` は単一ノード（1台のマシン）での実行を意味する。

#### posttrain_ddp.sh - ファインチューニング

```
scripts/posttrain_ddp.sh
```

事前学習済みモデル（`model_0030.pth`）を読み込み、Planning Decoderを有効にしてファインチューニングを実行する。
構成はpretrain_ddp.shとほぼ同じだが、以下の追加設定がある。

- `load_file=outputs/local_training/pretrain/model_0030.pth`: 事前学習のチェックポイント
- `use_planning_decoder=true`: Planning Decoderの有効化

学習トピック: 2段階学習（Pretrain → Posttrain）

LEADは2段階学習を採用している。
第1段階（Pretrain）: センサー入力の特徴量抽出を学習する。カメラ画像やLiDAR点群の理解を重点的に行う。
第2段階（Posttrain）: Planning Decoderを追加し、軌道計画タスクのファインチューニングを行う。
この2段階アプローチにより、安定した特徴量表現の上に計画能力を構築できる。

### 2.3 データ準備スクリプト

#### build_cache.py - PyTorchデータキャッシュの構築

```
python scripts/build_cache.py
```

学習データをPyTorchの永続キャッシュとして事前構築する。
`CARLAData` データセットクラスに `build_cache=True` を渡し、全データを1回走査してキャッシュファイルを生成する。
CPUコア数と同じワーカー数で並列処理を行い、I/Oボトルネックを軽減する。

#### build_buckets_pretrain.py / build_buckets_posttrain.py - データバケットの準備

```
python scripts/build_buckets_pretrain.py
python scripts/build_buckets_posttrain.py
```

学習データを「バケット」と呼ばれる単位に分割・整理する。
バケット化により、類似したデータを効率的にバッチ化し、学習の効率を向上させる。

### 2.4 評価スクリプト

#### eval_bench2drive.sh - Bench2Driveベンチマーク評価

```
scripts/eval_bench2drive.sh
```

Bench2Drive形式の短いルートでモデルを評価するスクリプト。
チェックポイントディレクトリとルートファイルを指定し、Bench2Driveリーダーボードの評価器を起動する。

主な環境変数:
- `CHECKPOINT_DIR`: モデルのチェックポイントディレクトリ
- `ROUTES`: 評価ルートのXMLファイル
- `IS_BENCH2DRIVE=1`: Bench2Driveモードの指定
- `PLANNER_TYPE=only_traj`: 軌道ベースの計画器を使用

#### eval_longest6.sh / eval_town13.sh - その他のベンチマーク

- `eval_longest6.sh`: Longest6ベンチマーク（中距離ルート）の評価
- `eval_town13.sh`: Town13ベンチマーク（長距離ルート、未知の町での汎化性能テスト）の評価

学習トピック: 主要ベンチマーク

Bench2Drive: 220本の短いルートで構成されるベンチマーク。基本的な運転能力を幅広く評価する。
Longest6: 6つの長めのルートで構成。長距離走行での安定性を測定する。
Town13: 学習データに含まれない未知の町（Town13）でのテスト。モデルの汎化性能を評価する重要なベンチマーク。

#### eval_expert.sh / eval_expert_123d.sh - エキスパートデータ収集

```
scripts/eval_expert.sh
scripts/eval_expert_123d.sh
```

エキスパートエージェント（ルールベースの運転者）を使用してデータを収集するスクリプト。
エキスパートは CARLA の地図情報を直接参照して最適な運転行動を計算する。

eval_expert.sh は Autopilot リーダーボードを使用し、`lead/expert/expert.py` をエージェントとして実行する。
eval_expert_123d.sh は Py123D版のエキスパートを使用する。

学習トピック: イミテーション学習とエキスパートデータ

E2E自動運転では、エキスパート（熟練ドライバー）の運転データからモデルを学習する「イミテーション学習」が基本的な手法。
CARLAのエキスパートは地図情報やシミュレータの内部状態にアクセスできるため、人間の運転データを使わずに
大量の高品質なデータを自動生成できる。

### 2.5 ユーティリティスクリプト

#### download_one_checkpoint.sh / download_one_route.sh

動作確認用に、1つのチェックポイントまたは1つのルートデータを素早くダウンロードするスクリプト。
初めてプロジェクトを触る際の簡易テストに便利。

#### unzip_routes.sh - 並列解凍

```
scripts/unzip_routes.sh
```

`data/carla_leaderboard2/zip/` 以下のZIPファイルを GNU parallel を使って最大64プロセスで並列解凍する。
大量のルートデータを効率的に展開するためのスクリプト。
SLURMジョブ設定（1日、8CPU、64GBメモリ）も含まれている。

#### random_free_port.sh

空いているポート番号をランダムに取得するユーティリティ。
複数のCARLAインスタンスを同時に起動する際のポート衝突を回避するために使用する。

### 2.6 main.sh - 環境変数の設定

```
source scripts/main.sh
```

プロジェクト全体で使用する環境変数を設定するスクリプト。他のスクリプトから `source` で読み込んで使う。

設定内容:
- `CARLA_ROOT`: CARLAのインストールパス（`3rd_party/CARLA_0915`）
- `PYTHONPATH`: CARLAのPython APIパスを追加
- `PATH`: プロジェクトルートと `scripts/` をPATHに追加
- NavSim関連: `NUPLAN_MAP_VERSION`、`NAVSIM_DEVKIT_ROOT` など
- Py123D関連: `PY123D_DATA_ROOT`

### 2.7 tools/ - 追加ツール

`scripts/tools/` には以下のツールが含まれる。

- `Dockerfile.master` / `make_docker.sh` / `run_docker.sh`: Docker環境の構築と実行
- `result_parser.py`: 評価結果のパース
- `route_bridge.py` / `route_bridge.sh`: ルートファイルの変換・ブリッジ
- `proxy_simulator/`: CARLAのプロキシシミュレータ（マップデータの前処理）

---

## 3. slurm/ - クラスタインフラストラクチャ

`slurm/` ディレクトリには、SLURMクラスタ上で学習・評価を実行するためのインフラコードが格納されている。

学習トピック: SLURMとは

SLURMは高性能計算（HPC）クラスタで広く使われるジョブスケジューラ。
GPUやCPUなどの計算資源をユーザー間で公平に配分し、ジョブのキューイング・スケジューリングを行う。
`sbatch` コマンドでジョブを投入し、`#SBATCH` ディレクティブでリソース要件を指定する。

### 3.1 init.sh - ジョブ管理関数

```
source slurm/init.sh
```

全ての実験スクリプトの基盤となるシェルスクリプト。実験管理のための関数と変数を定義する。

#### 自動生成されるID・ディレクトリ

- `EXPERIMENT_NAME`: ディレクトリ名から取得（例: `001_example`）
- `SCRIPT_NAME`: スクリプトファイル名から取得（例: `000_pretrain1_0`）
- `SLURM_JOB_DATE`: 現在日時（`YYMMDD_HHMMSS` 形式）
- `EXPERIMENT_RUN_ID`: 上記を結合した一意なID（最大64文字）
- `EXPERIMENT_RUN_DIR`: 出力ディレクトリのパス構造
- `EXPERIMENT_SEED`: スクリプトファイル名の末尾の数字をシードとして使用

出力ディレクトリ:
- 学習: `outputs/training/{EXPERIMENT_RUN_DIR}`
- 評価: `outputs/evaluation/{EXPERIMENT_RUN_DIR}`

#### 提供される関数

resume(checkpoint_dir):

既存のチェックポイントから学習を再開する。最新の `model_*.pth` ファイルを自動検出し、WandBの実行IDも引き継ぐ。

posttrain(checkpoint_file):

事前学習済みモデルからファインチューニングを開始する。
引数がディレクトリの場合は最新のモデルファイルを自動検出。
`continue_epoch=false` を設定し、エポック番号は0からリスタートする。

train():

学習ジョブを投入する。SLURMが利用可能な場合は `sbatch` で投入し、利用不可の場合はローカルで直接実行する。
WandBのID、シード、説明、ログディレクトリを自動設定する。

evaluate():

モデル評価を実行する。`EVALUATION_DATASET`（bench2drive、Town13、longest6）に応じてデータセットを選択。
チェックポイントの存在確認を行い、`screen` セッションで `slurm/evaluate.sh` をバックグラウンド実行する。

evaluate_bench2drive() / evaluate_town13() / evaluate_longest6():

各ベンチマーク用のラッパー関数。データセット名とタイムアウト設定を適切にセットした上で `evaluate()` を呼び出す。
Town13は3日間、Longest6は10時間のタイムアウトが設定されている。

evaluate_expert() / evaluate_expert_bench2drive() / evaluate_expert_town13() / evaluate_expert_longest6():

エキスパートエージェントの評価用関数群。モデルの代わりにエキスパートを実行する。

### 3.2 train.sh - SLURM学習ジョブ

SLURMジョブのリソース要件:

```
ノード数: 1
GPU: 1（スクリプト内で自動検出、実際には4GPU使用を想定）
CPU: 8コア
メモリ: 256GB
制限時間: 3日間
パーティション: a100-galvani
```

処理の流れ:
1. SLURMジョブ情報の表示
2. Conda環境（`lead`）の有効化
3. GPU数の自動検出（ローカル実行時は1GPU）
4. 環境変数の設定（NCCL、OMP、CUDAアロケータ）
5. GPU数に応じて `torchrun`（マルチGPU）または `python`（シングルGPU）で学習を開始

### 3.3 evaluate.sh - 評価ジョブ

ルートごとに個別のSLURMジョブを生成し、並列で評価を実行する。
`slurm/evaluation/evaluate_scripts_generator.py` で各ルートのbashスクリプトを自動生成し、
`slurm/evaluation/evaluate.py` の `SlurmJobPool` で管理する。

### 3.4 config_slurm.py - SLURM環境変数の読み取り

SLURM環境変数（`SLURM_JOB_ID`、`SLURM_CPUS_PER_TASK` など）を読み取り、
Pythonコードから参照できるようにする設定モジュール。

### 3.5 evaluation/ - 評価サブシステム

#### evaluate.py - SlurmJobPool

SLURMジョブプールを管理するクラス。

主な機能:
- 指定されたディレクトリ内のbashスクリプトを収集・ソート
- IDリストによるフィルタリング（特定のルートだけ評価可能）
- ジョブプールのサイズを制限して並列度を管理
- リトライロジック（失敗したルートの再実行）
- WandBへのメトリクスログ

#### evaluate_scripts_generator.py - ルートごとのスクリプト生成

各ルートXMLファイルに対応するSLURMジョブスクリプトを自動生成する。
現在の環境変数を全てスクリプト内にエクスポートし、再現性を確保する。
エキスパートモードとモデル評価モードで異なるリーダーボード評価器を選択する。

#### evaluate_utils.py - ユーティリティ

- クラッシュの検出
- ジョブステータスの確認
- 評価結果の集計

#### evaluate_wandb_logger.py - WandBログ

評価メトリクスをWandB（Weights & Biases）にログする。
ルート数に基づいてログ設定を初期化する。

#### merge_route_json.py - ルート結果の結合

個別ルートの評価結果JSONを1つのファイルに統合する。

### 3.6 data_collection/ - データ収集

#### collect_data.py

並列データ収集ジョブを生成するスクリプト。
複数のCARLAインスタンスとルートを同時に実行し、エキスパートデータを効率的に収集する。

#### delete_failed_routes.py

失敗したルートのデータを削除するクリーンアップスクリプト。

#### print_collect_data_progress.py

データ収集の進捗状況を表示するモニタリングツール。

### 3.7 configs/ - 設定ファイル

テキストファイルベースの設定で、データセットごとにパラメータを管理する。

最大リトライ回数（max_num_attempts_*.txt）:

各データセット（Town13、bench2drive220、longest6、fail2drive210、collect_data）ごとに
ルート評価の最大試行回数を定義する。CARLAの不安定性やランダム性により、一部のルートが
失敗することがあるため、リトライ機構が重要になる。

最大並列ジョブ数（max_num_parallel_jobs_*.txt）:

クラスタリソースを効率的に使うために、同時に実行するジョブの上限を設定する。

その他:
- `max_sleep.txt`: ジョブ間の待機時間
- `wandb_log_frequency_training_images.txt`: 学習画像のWandBログ頻度
- `wandb_log_frequency_training_scalar.txt`: スカラー値のWandBログ頻度

### 3.8 experiments/001_example/ - 完全な実験ワークフロー

`001_example` ディレクトリは、事前学習から評価までの完全なワークフローを示すテンプレートである。
ファイル名の先頭3桁が実行順序を表し、末尾の数字がランダムシードを表す。

#### ワークフローの流れ

ステップ1 - 事前学習:

```
000_pretrain1_0.sh
```

RegNetY-032アーキテクチャで事前学習を実行。4GPU、32CPU、3日間。

ステップ2 - ファインチューニング（3シード）:

```
010_postrain32_0.sh  (シード0)
011_postrain32_1.sh  (シード1)
012_postrain32_2.sh  (シード2)
```

事前学習のチェックポイントを読み込み、Planning Decoderを追加してファインチューニング。
3つの異なるシードで実行し、結果の統計的な安定性を確保する。

ステップ3 - Bench2Drive評価（3シード）:

```
020_b2d_0.sh / 021_b2d_1.sh / 022_b2d_2.sh
```

ステップ4 - Longest6評価（3シード）:

```
030_longest6_0.sh / 031_longest6_1.sh / 032_longest6_2.sh
```

ステップ5 - Town13評価（3シード）:

```
040_town13_0.sh / 041_town13_1.sh / 042_town13_2.sh
```

学習トピック: 実験の再現性とシード管理

機械学習では、ランダムシードの違いにより結果が変動する。
1つのシードだけで性能を評価すると、偶然の結果を過大/過小評価してしまう可能性がある。
LEADでは3つのシードで実験を繰り返し、平均と分散を報告することで統計的な信頼性を担保する。

---

## 4. notebooks/ - 分析ノートブック

`notebooks/` ディレクトリには、データの可視化・分析用のJupyter Notebookが格納されている。

### 4.1 carla_offline_inference.ipynb - オフライン推論の可視化

モデルのチェックポイントを読み込み、記録済みデータに対してオフラインで推論を実行する。
シミュレータを起動せずにモデルの予測結果を視覚的に確認できる。

用途:
- モデルの予測軌道と正解軌道の比較
- センサー入力に対するモデルの応答の確認
- デバッグやプレゼンテーション資料の作成

### 4.2 data_format.ipynb - データ形式の調査

LEADで使用するデータの形式・構造を調査するためのノートブック。
各フィールドの意味、データ型、値の範囲などを確認できる。

初学者がデータの構造を理解するのに適している。

### 4.3 inspect_expert_output.ipynb - エキスパート出力の可視化

エキスパートエージェントが生成したデータを可視化する。
カメラ画像、LiDAR点群、バウンディングボックス、走行軌跡などを確認できる。

学習トピック: エキスパートデータの構成

エキスパートデータには以下の情報が含まれる:
- 複数視点のカメラ画像（前方、左右、後方など）
- LiDAR 3D点群データ
- レーダーデータ
- 車両の位置・速度・加速度
- ステアリング・スロットル・ブレーキの操作量
- バウンディングボックス（周囲の車両・歩行者の位置）
- ナビゲーションコマンド（直進、左折、右折など）

### 4.4 inspect_sensor_agent_io.ipynb - センサーエージェントI/O分析

センサーエージェント（推論時のエージェント）の入出力を分析するノートブック。
モデルが受け取るセンサー入力と、出力する制御コマンドの関係を調べることができる。

---

## 5. lead/infraction_webapp/ - 違反分析Webダッシュボード

### 概要

Flaskベースの Webアプリケーション（ポート5000）で、CARLAの評価中に発生した違反（infraction）を
視覚的に分析するためのダッシュボード。

### APIエンドポイント

- `/`: メインのダッシュボード画面
- `/api/output_directories`: 評価出力ディレクトリの一覧
- `/api/routes`: ルート情報の取得
- `/api/infractions`: 違反データの取得（JSON形式）
- `/api/videos`: 走行動画の配信

### 主要機能

動画再生と違反タイムライン:

評価中に録画された走行動画を再生し、違反が発生したタイミングをタイムライン上に表示する。
違反の種類（衝突、赤信号無視、車線逸脱など）ごとに異なるマーカーで表示される。

キーボードナビゲーション:

フレーム単位でのコマ送り・コマ戻しが可能。違反発生の瞬間を詳細に分析できる。

レガシー形式の対応:

旧形式（リスト形式）と新形式（infractions + video_fps を含むオブジェクト形式）の両方をサポートする。

学習トピック: 自動運転における違反とペナルティ

CARLAの評価では、以下のような違反が検出・記録される:
- 他車両との衝突
- 歩行者との衝突
- 赤信号の無視
- 一時停止標識の無視
- 車線逸脱
- ルートからの逸脱

各違反にはペナルティスコアが設定されており、最終的な Driving Score に影響する。
このWebダッシュボードを使って違反の原因を分析し、モデルの改善に活用する。

---

## 6. lead/visualization/visualizer.py - 可視化モジュール

### 概要

学習データ、モデルの予測結果、グラウンドトゥルースを統合的に可視化するためのクラス。

### Visualizer クラス

初期化パラメータ:
- `config`: 学習設定（TrainingConfig）
- `data`: 入力データ辞書
- `prediction`: モデルの予測結果（Prediction / ClosedLoopPrediction / OpenLoopPrediction）
- `training`: 学習時の可視化かどうか
- `config_test_time`: テスト時の設定（ClosedLoopConfig）
- `test_time`: テスト時の可視化かどうか

### 可視化項目

`visualize_sample()` メソッドで以下の情報を1つの画像にまとめて可視化する:

- 複数視点のカメラ画像（前方、左右、後方）
- BEV（Bird's Eye View）セマンティックマップ
- バウンディングボックス（3Dオブジェクト検出結果）
- LiDAR点群の投影
- レーダーデータの投影
- 予測軌道 vs 正解軌道
- ナビゲーションコマンド

学習トピック: BEV（鳥瞰図）表現

BEVは自動運転で広く使われる表現形式。3Dセンサーデータを上方からの2D投影に変換し、
車両周囲の環境をマップのように表現する。道路構造、車両位置、歩行者位置などを
直感的に把握でき、計画モジュールの入力として適している。

---

## 7. プロジェクト環境設定

### pyproject.toml

プロジェクトの基本設定:
- プロジェクト名: lead
- バージョン: 0.0.1
- Python要件: 3.10以上
- リンター: ruff
- 型チェッカー: basedpyright

### requirements.txt

333パッケージの依存関係を管理。主要なパッケージ:
- torch 2.5.0: 深層学習フレームワーク
- carla 0.9.15: CARLAシミュレータのPython API
- ray 2.47.1: 分散計算フレームワーク
- wandb: 実験管理・ログ
- flask: Webダッシュボード
- opencv-python / PIL: 画像処理
- matplotlib: グラフ描画

### environment.yml

Conda環境設定ファイル。Python 3.10.15 を使用する `lead` 環境を定義する。

```
conda env create -f environment.yml
conda activate lead
```

### main.sh - 環境変数の一括設定

前述の `scripts/main.sh` を `source` で読み込むことで、CARLA、NavSim、Py123D の各種パスが設定される。
プロジェクトのルートディレクトリで以下を実行する:

```
export LEAD_PROJECT_ROOT=$(pwd)
source scripts/main.sh
```

学習トピック: 環境構築の全体像

LEADプロジェクトのセットアップ手順:
1. リポジトリのクローン
2. Conda環境の作成（`environment.yml`）
3. 依存パッケージのインストール（`requirements.txt`）
4. CARLAのダウンロード（`scripts/setup_carla.sh`）
5. 環境変数の設定（`scripts/main.sh`）
6. データの取得（`scripts/download_one_route.sh` で動作確認）
7. データキャッシュの構築（`scripts/build_cache.py`）

---

## 全体アーキテクチャの関係図

以下は、各コンポーネント間の関係を示す概念図である。

```
[scripts/setup_carla.sh] --> CARLA 0.9.15 のインストール
[scripts/start_carla.sh] --> CARLA サーバーの起動
                                    |
                                    v
[scripts/eval_expert.sh] -----> [lead/leaderboard_wrapper.py] -----> エキスパートデータ収集
                                    |
                                    v
[scripts/build_cache.py] -----> データキャッシュ構築
[scripts/build_buckets_*.py] -> データバケット構築
                                    |
                                    v
[scripts/pretrain_ddp.sh] ----> 事前学習（DDP）
[scripts/posttrain_ddp.sh] ---> ファインチューニング
                                    |
                                    v
[scripts/eval_bench2drive.sh] -> ベンチマーク評価
                                    |
                                    v
[lead/infraction_webapp/] ----> 違反分析Webダッシュボード
[lead/visualization/] --------> 可視化ツール
[notebooks/] ------------------> 分析ノートブック
```

SLURMクラスタ上では、上記のフローが `slurm/init.sh` の関数群を通じて自動化される。
`slurm/experiments/001_example/` がその完全な実装例である。

---

## まとめ

このドキュメントでは、LEADプロジェクトのスクリプト・ツール群を解説した。
主要なポイントを振り返る:

- CARLAの環境管理（setup / start / clean / reset）は `scripts/` で提供される
- ローカルでの学習・評価は `scripts/` の各シェルスクリプトで直接実行可能
- クラスタ環境では `slurm/init.sh` の関数群を使って実験を管理する
- 2段階学習（Pretrain → Posttrain）と3シードによる統計的な実験設計が標準
- `notebooks/` でデータとモデルの挙動を視覚的に分析できる
- `infraction_webapp` で違反の詳細分析がブラウザ上で可能
- `leaderboard_wrapper.py` が評価の統一インターフェースとして機能する
