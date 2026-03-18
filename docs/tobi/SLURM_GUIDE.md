# SLURM ガイド - 基礎知識からLEADでの活用まで

このドキュメントは、SLURMを初めて使う人が基礎概念を理解し、LEADプロジェクトでのSLURMの使い方を把握するためのガイドです。

---

## Part 1: SLURMの基礎知識

### SLURMとは何か

SLURM（Simple Linux Utility for Resource Management）は、HPCクラスタ向けのオープンソースジョブスケジューラです。Top-500スーパーコンピュータの50%以上で採用されており、クラスタ管理のデファクトスタンダードとなっています。

SLURMが解決する問題:
数百人のユーザが限られたGPU、CPU、メモリを共有するHPC環境では、誰かがリソースを公平に割り当てる仕組みが必要です。SLURMはこの役割を自動的に担います。

SLURMの3つの主要機能:

| 機能 | 説明 |
|------|------|
| リソース割り当て | 計算ノードのCPU、GPU、メモリをユーザに排他的/共有的に割り当てる |
| ジョブ実行 | 割り当てられたノード上でユーザのプログラムを起動・管理する |
| キュー管理 | 待ちジョブの優先度を決定し、リソースが空き次第実行する |

アーキテクチャ:

```
slurmctld（管理デーモン）
  - 管理ノード上で動作
  - ジョブスケジューリング、リソース監視を担当
  - クラスタ全体の状態を把握

slurmd（計算ノードデーモン）
  - 各計算ノード上で動作
  - slurmctldからの指示でジョブを起動・監視
  - ノードのリソース状態を報告
```

### コア概念

SLURMを理解するうえで押さえるべき基本概念を以下にまとめます。

- ノード（Node）: 物理マシンまたは仮想マシン。GPU、CPU、メモリを持つ計算の実行単位。
- パーティション（Partition）: ノードの論理グループで、ジョブキューとして機能する。用途別にパーティションが分かれることが多い。
  - 例: `gpu`（GPU計算用）、`cpu`（CPU計算用）、`debug`（デバッグ用、短時間制限）、`long`（長時間ジョブ用）
- ジョブ（Job）: SLURMにおけるリソース割り当ての単位。1つのジョブに対して指定されたリソースが確保される。
- ジョブステップ（Job Step）: ジョブ内の個別の処理単位。`srun`で起動される。1つのジョブ内に複数のジョブステップを含められる。
- タスク（Task）: 個別のプロセス。`--ntasks`で指定する。MPIなどの並列処理で複数タスクを起動する場合に使う。
- ジョブ配列（Job Array）: 類似したジョブを一括投入する仕組み。各ジョブは`SLURM_ARRAY_TASK_ID`で区別される。パラメータスイープなどで便利。

概念の階層関係:
```
パーティション
  └── ジョブ
        └── ジョブステップ（srun）
              └── タスク（プロセス）
```

### 主要コマンド

SLURMの日常操作で使うコマンド一覧です。

| コマンド | 用途 | 使用例 |
|----------|------|--------|
| `sbatch` | バッチスクリプトをキューに投入（非対話） | `sbatch train.sh` |
| `srun` | 並列ジョブを実行（対話的 or sbatch内で使用） | `srun --gres=gpu:1 python train.py` |
| `salloc` | リソースを対話的に確保（シェルが起動する） | `salloc --gres=gpu:2 --time=1:00:00` |
| `squeue` | ジョブキューを確認 | `squeue -u $USER` |
| `scancel` | ジョブをキャンセル | `scancel 12345` |
| `sinfo` | クラスタ/パーティションの状態を確認 | `sinfo -p gpu` |
| `sacct` | 完了済みジョブの履歴・リソース使用量を確認 | `sacct -j 12345 --format=JobID,MaxRSS,Elapsed` |

squeueの主なステータス:
- PD（Pending）: リソース待ちでキューに並んでいる状態
- R（Running）: 実行中
- CG（Completing）: 終了処理中
- CA（Cancelled）: キャンセルされた

```bash
# 自分のジョブだけを表示
squeue -u $USER

# 特定パーティションのジョブを表示
squeue -p a100-galvani

# ジョブの詳細情報を表示
scontrol show job 12345
```

### SBATCHディレクティブ

SBATCHスクリプトの先頭に `#SBATCH` で記述するディレクティブ一覧です。これらはジョブのリソース要求やメタ情報を指定します。

| ディレクティブ | 説明 | 例 |
|----------------|------|-----|
| `--partition` | 投入先パーティション | `--partition=a100-galvani` |
| `--nodes` | 使用ノード数 | `--nodes=1` |
| `--ntasks` | 総タスク数（プロセス数） | `--ntasks=1` |
| `--ntasks-per-node` | ノードあたりのタスク数 | `--ntasks-per-node=1` |
| `--cpus-per-task` | タスクあたりのCPUコア数 | `--cpus-per-task=8` |
| `--gres` | 汎用リソース（GPUなど）の要求 | `--gres=gpu:a100:4` |
| `--gpus-per-node` | ノードあたりのGPU数 | `--gpus-per-node=4` |
| `--mem` | ノードあたりのメモリ上限 | `--mem=256gb` |
| `--time` | 最大実行時間（D-HH:MM:SS形式） | `--time=3-00:00:00` |
| `--job-name` | ジョブ名（squeueで表示される） | `--job-name=my_training` |
| `--output` | 標準出力ファイル（%jがジョブIDに置換） | `--output=logs/%j.out` |
| `--error` | 標準エラー出力ファイル | `--error=logs/%j.err` |
| `--array` | ジョブ配列の範囲指定 | `--array=0-99` |
| `--mail-type` | メール通知のトリガー | `--mail-type=FAIL,END` |
| `--mail-user` | 通知先メールアドレス | `--mail-user=user@example.com` |
| `--exclusive` | ノードを排他的に使用 | `--exclusive` |
| `--dependency` | ジョブ間の依存関係 | `--dependency=afterok:12345` |

スクリプト例:
```bash
#!/bin/bash
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256gb
#SBATCH --time=3-00:00:00
#SBATCH --job-name=lead_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=user@example.com

# ここにジョブの処理を記述
python train.py
```

### GPU割り当ての仕組み

SLURMはGRES（Generic Resource）サブシステムを通じてGPUを管理します。

GPUの要求方法:
```bash
# GPU 2枚を要求（種類は問わない）
--gres=gpu:2

# A100を4枚指定して要求
--gres=gpu:a100:4

# ノードあたりのGPU数で指定
--gpus-per-node=4
```

SLURMがGPUを割り当てると、`CUDA_VISIBLE_DEVICES`環境変数が自動的に設定されます。これにより、ジョブは割り当てられたGPUのみを参照します。

```bash
# 例: 4GPUのうち2, 3番が割り当てられた場合
# CUDA_VISIBLE_DEVICES=2,3 が自動設定される
# プログラムからはデバイス0, 1として見える
```

ソケットアフィニティ:
SLURMはGPUとCPUが同一ソケット（NUMAノード）に属するように割り当てを最適化します。これにより、GPU-CPU間のデータ転送のレイテンシが最小化されます。

### SLURM環境変数

ジョブ内で利用可能な主要な環境変数です。これらはSLURMが自動的に設定します。

| 環境変数 | 説明 |
|----------|------|
| `SLURM_JOB_ID` | ジョブの一意なID |
| `SLURM_JOB_NAME` | ジョブ名（--job-nameで指定した値） |
| `SLURM_JOB_NODELIST` | 割り当てられたノード一覧 |
| `SLURM_NNODES` | 割り当てられたノード数 |
| `SLURM_NTASKS` | 総タスク数 |
| `SLURM_PROCID` | グローバルランク（全ノード通しでのプロセス番号） |
| `SLURM_LOCALID` | ローカルランク（ノード内でのプロセス番号） |
| `SLURM_CPUS_PER_TASK` | タスクあたりのCPU数 |
| `SLURM_ARRAY_JOB_ID` | ジョブ配列の親ジョブID |
| `SLURM_ARRAY_TASK_ID` | ジョブ配列内のタスクインデックス |

使用例:
```bash
# OpenMPスレッド数をSLURMの割り当てに合わせる
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ジョブIDを含むファイル名でログを保存
echo "Training log" > log_${SLURM_JOB_ID}.txt
```

### 分散学習との連携

SLURMで分散深層学習を行う場合、2つのパターンがあります。

パターンA: srunが直接プロセスを起動
```bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2

srun python train.py
```
- srunが各ノードに指定数のプロセスを起動する
- `SLURM_PROCID`がそのままグローバルランクとして使える
- MPI互換の起動方法

パターンB: srun + torchrun（各ノードに1つtorchrunを起動）
```bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2

srun torchrun --nproc_per_node=4 train.py
```
- `ntasks-per-node=1`が重要（各ノードで1つだけtorchrunを起動）
- torchrunが各ノード内のGPUプロセスを管理する

LEADはパターンBを採用しています。ただし、現在は単一ノード（`--nodes=1`）での学習を前提としているため、`torchrun --standalone`で起動しています。

torchrunの役割:
- 指定数のプロセスを起動する
- `LOCAL_RANK`、`RANK`、`WORLD_SIZE`環境変数を自動設定する
- c10dバックエンドによるランデブー（プロセス間の初期接続確立）を行う

NCCL設定:
分散学習で使用するNCCL（NVIDIA Collective Communications Library）には環境依存の設定が必要な場合があります。
- `NCCL_P2P_DISABLE=1`: P2P通信を無効化（一部のハードウェアで必要）
- `NCCL_P2P_LEVEL=NVL`: NVLink経由のP2P通信レベルを指定

### ML/DLワークロードの一般パターン

HPCクラスタ上で機械学習/深層学習を行う際の一般的なパターンです。

チェックポイント（時間制限対策）:
SLURMジョブには時間制限があるため、長時間の学習では定期的にモデルの状態を保存し、ジョブが終了しても途中から再開できるようにします。
```bash
# チェックポイントから再開するジョブ投入
sbatch --dependency=afterok:$JOB1 resume_training.sh
```

ジョブチェイン:
前のジョブが成功したら次のジョブを自動投入する仕組みです。
```bash
JOB1=$(sbatch --parsable pretrain.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 finetune.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 evaluate.sh)
```

データローダのCPU確保:
PyTorchのDataLoaderは`num_workers`で指定した数のCPUワーカーを使うため、`--cpus-per-task`を十分に確保する必要があります。

ローカルSSDへのデータコピー:
ネットワークファイルシステムからの読み込みはボトルネックになりやすいため、ジョブ開始時にデータをノードのローカルSSDにコピーしてから学習を開始するパターンがよく使われます。

---

## Part 2: LEADにおけるSLURMの使い方

### 設計思想

LEADのSLURMスクリプトは「1実験 = 1 bashスクリプト」という原則に基づいています。全てのスクリプトはGitでバージョン管理され、実験の再現性を保証します。

命名規則:
```
slurm/experiments/<exp_id>_<exp_name>/<step_id>_<step_name>_<seed>.sh
```

例:
```
slurm/experiments/001_example/000_pretrain1_0.sh
                               ^^^            ^
                               ステップID      シード値
```

- `exp_id`: 実験の通し番号（001, 002, ...）
- `exp_name`: 実験の名前（example, baseline, ...）
- `step_id`: 実験内のステップ番号（000=事前学習, 010=事後学習, 020=評価, ...）
- `step_name`: ステップの名前（pretrain1, postrain32, b2d, ...）
- `seed`: 乱数シード値（0, 1, 2, ...）。スクリプト名の最後の `_` 以降が自動的にシードとして抽出される。

### ディレクトリ構成

```
slurm/
├── init.sh               # ヘルパー関数定義（train, resume, evaluate等）
├── train.sh              # 学習ジョブスクリプト（SBATCHディレクティブ付き）
├── evaluate.sh           # 評価ジョブスクリプト（スクリプト生成 + ジョブプール起動）
├── evaluate_expert.sh    # エキスパートエージェント評価
├── config_slurm.py       # SLURM環境変数の型付きアクセス（Pythonクラス）
├── configs/              # テキストファイルで設定値を管理
│   ├── max_num_attempts_*.txt       # データセット別の最大再試行回数
│   ├── max_num_parallel_jobs_*.txt  # データセット別の最大並列ジョブ数
│   ├── max_sleep.txt                # CARLA起動待ち時間
│   └── wandb_log_frequency_*.txt    # WandBログ頻度
├── evaluation/           # 評価ジョブ管理
│   ├── evaluate.py                  # SlurmJobPool（ジョブプール管理クラス）
│   ├── evaluate_scripts_generator.py # ルートごとのSBATCHスクリプト生成
│   ├── evaluate_utils.py            # ジョブ投入・監視ユーティリティ
│   ├── evaluate_wandb_logger.py     # WandBへのリアルタイムログ
│   └── merge_route_json.py          # 全ルートの結果をmerged.jsonに集約
├── data_collection/      # データ収集ジョブ管理
│   ├── collect_data.py              # ルートごとにSLURMスクリプト生成・投入
│   ├── delete_failed_routes.py      # 失敗ルートのデータ削除
│   └── print_collect_data_progress.py # 収集進捗表示
└── experiments/          # 実験スクリプト
    └── 001_example/      # 実験例（13個のスクリプト）
```

### init.sh のヘルパー関数

`init.sh`は全ての実験スクリプトの冒頭で`source`され、共通の環境変数とヘルパー関数を提供します。

自動設定される環境変数:

| 変数名 | 説明 | 例 |
|--------|------|-----|
| `EXPERIMENT_NAME` | スクリプトの親ディレクトリ名 | `001_example` |
| `SCRIPT_NAME` | スクリプトのファイル名（拡張子なし） | `000_pretrain1_0` |
| `SLURM_JOB_DATE` | ジョブ投入時のタイムスタンプ | `251018_092144` |
| `EXPERIMENT_RUN_ID` | 一意な実行ID（64文字以内） | `001_example_000_pretrain1_0_251018_092144` |
| `EXPERIMENT_RUN_DIR` | 出力ディレクトリのサブパス | `001_example/000_pretrain1_0/251018_092144` |
| `EXPERIMENT_SEED` | スクリプト名末尾から抽出されたシード | `0` |

主要関数:

- `train()`: SBATCHでジョブを投入する。SLURMが検出されない場合はローカル実行にフォールバックする。追加のSBATCHオプションを引数で渡せる。
  ```bash
  train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
  ```

- `resume()`: チェックポイントディレクトリを指定して学習を再開する。最新のmodel_*.pthを自動検出し、WandB IDも引き継ぐ。optimizer、scheduler、scalerの状態が全て復元される。
  ```bash
  resume outputs/training/001_example/000_pretrain1_0/251018_092144
  ```

- `posttrain()`: 事前学習済みモデルからファインチューニングを開始する。チェックポイントファイルまたはディレクトリを指定できる（ディレクトリの場合は最新のmodel_*.pthを自動検出）。
  ```bash
  posttrain outputs/training/001_example/000_pretrain1_0/251018_092144
  ```

- `evaluate_bench2drive()`: Bench2Driveベンチマークで評価を実行する。preemptableパーティションを使用。
- `evaluate_longest6()`: Longest6ベンチマークで評価を実行する。タイムアウト10時間。
- `evaluate_town13()`: Town13ベンチマークで評価を実行する。タイムアウト3日。

### 学習ジョブ (train.sh)

`train.sh`はSBATCHディレクティブを含む学習用スクリプトです。

デフォルトのSBATCHディレクティブ:
```bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mem=256gb
```

注意: init.shの`train()`関数から呼び出される際、`--gres=gpu:4`などのオプションが上書きされます。SBATCHディレクティブよりコマンドライン引数が優先されるためです。

処理の流れ:
1. SLURM環境であればジョブ情報を表示（`scontrol show job`）
2. Conda環境をアクティベート
3. OMP_NUM_THREADSをSLURMの割り当てCPU数に設定
4. NCCL最適化設定を適用
   - `NCCL_P2P_DISABLE=1`
   - `NCCL_P2P_LEVEL=NVL`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（メモリ断片化の軽減）
5. GPU数を自動検出（`nvidia-smi`でクエリ）
6. GPU数に応じてtorchrunまたは直接起動
   - 1 GPU: `python lead/training/train.py`
   - 複数GPU: `torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node lead/training/train.py`

### 評価パイプライン

LEADの評価はCARLAシミュレータ上で行われ、ルートごとに独立したSLURMジョブとして実行されます。3つのステージで構成されます。

ステージ1: スクリプト生成（evaluate_scripts_generator.py）

`evaluate.sh`から呼び出され、ルートフォルダ内の各XMLルートファイルに対応するbashスクリプトを生成します。

各スクリプトに含まれる内容:
- SBATCHディレクティブ（パーティション、GPU、メモリ、タイムアウト）
- 全環境変数のエクスポート
- CARLAサーバの起動（`CarlaUE4.sh`をバックグラウンド起動 + 180秒待機）
- ポートのランダム割り当て（`random_free_port.sh`でポート衝突を回避）
- CARLAのクリーンアップ処理
- 評価スクリプトの実行

ステージ2: ジョブプール管理（evaluate.py の SlurmJobPool）

生成されたスクリプトをジョブプールとして管理し、並列実行します。

動作:
- `max_parallel_jobs`（設定ファイルで指定）まで並列にジョブを投入
- 5秒ごとにジョブのステータスをチェック
- ジョブが終了したら結果JSONを検証
- 失敗したジョブは`max_num_attempts`まで自動再投入
- 10回を超える再試行ではpreemptableパーティション（A100）に切り替え
- WandBにルートごとの結果をリアルタイムでログ
- 全ルート完了後にメトリクスを集約

ステージ3: 結果集約（merge_route_json.py）

全ルートの結果JSONファイルを`merged.json`に集約します。集約されるメトリクス:
- driving score（運転スコア）
- success rate（成功率）
- eval num（評価済みルート数）

各データセットのルート総数:
- Town13: 20ルート
- longest6: 36ルート
- bench2drive: 220ルート

### データ収集パイプライン

エキスパートエージェントを使ってCARLAシミュレータ上でデータを収集するパイプラインです。

処理の流れ:
1. `collect_data.py`がルートフォルダ内のXMLファイルを読み込む
2. ルートごとにSLURMスクリプトを生成
3. `max_num_parallel_jobs`（48）まで並列にジョブを投入
4. 完了済みルートはスキップ
5. 20秒ごとにジョブの状態を確認し、失敗ジョブを再投入

エラーパターン検出と自動回復:
ログファイルの最終行を監視し、以下のパターンを検出するとscancelで強制終了し、再投入します。
- `Actor ... not found!`: CARLAのアクタが消失
- `Watchdog exception - Timeout`: タイムアウト
- `Engine crash handling finished; re-raising signal 11`: CARLAエンジンのクラッシュ

シナリオフィルタリング:
- ブラックリスト: `YieldToEmergencyVehicle`（不安定なため除外）
- シナリオタイプごとの最大ルート数制限（デフォルト40）

### 設定ファイル (configs/)

設定はテキストファイルに整数値1つだけを記載するシンプルな形式です。ジョブの実行中でも編集可能で、次回の読み込み時に反映されます。

| ファイル | 値 | 説明 |
|----------|-----|------|
| `max_num_attempts_Town13.txt` | 5 | Town13評価の最大再試行回数 |
| `max_num_attempts_bench2drive220.txt` | 6 | Bench2Drive評価の最大再試行回数 |
| `max_num_attempts_longest6.txt` | 6 | Longest6評価の最大再試行回数 |
| `max_num_attempts_collect_data.txt` | 1 | データ収集の最大再試行回数 |
| `max_num_parallel_jobs_Town13.txt` | 8 | Town13評価の最大並列ジョブ数 |
| `max_num_parallel_jobs_bench2drive220.txt` | 8 | Bench2Drive評価の最大並列ジョブ数 |
| `max_num_parallel_jobs_longest6.txt` | 8 | Longest6評価の最大並列ジョブ数 |
| `max_num_parallel_jobs_collect_data.txt` | 48 | データ収集の最大並列ジョブ数 |
| `max_sleep.txt` | 180 | CARLA起動後の待機時間（秒） |
| `wandb_log_frequency_training_images.txt` | - | 学習画像のWandBログ頻度 |
| `wandb_log_frequency_training_scalar.txt` | - | 学習スカラー値のWandBログ頻度 |

### 完全なワークフロー例 (001_example)

`slurm/experiments/001_example/`には、事前学習から評価までの全工程が13個のスクリプトで定義されています。

```
000_pretrain1_0.sh        # 事前学習（4GPU, a100, 3日）
010_postrain32_0.sh       # 事後学習 シード0（4GPU, L40S, 4日）
011_postrain32_1.sh       # 事後学習 シード1
012_postrain32_2.sh       # 事後学習 シード2
020_b2d_0.sh              # Bench2Drive評価 シード0
021_b2d_1.sh              # Bench2Drive評価 シード1
022_b2d_2.sh              # Bench2Drive評価 シード2
030_longest6_0.sh         # Longest6評価 シード0
031_longest6_1.sh         # Longest6評価 シード1
032_longest6_2.sh         # Longest6評価 シード2
040_town13_0.sh           # Town13評価 シード0
041_town13_1.sh           # Town13評価 シード1
042_town13_2.sh           # Town13評価 シード2
```

各スクリプトの構成例:

事前学習（000_pretrain1_0.sh）:
```bash
source slurm/init.sh
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
```

事後学習（010_postrain32_0.sh）:
```bash
source slurm/init.sh
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144
train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

評価（020_b2d_0.sh）:
```bash
source slurm/init.sh
export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327
evaluate_bench2drive
```

タイムラインの概算:

| ステージ | 所要時間 | 備考 |
|----------|----------|------|
| 事前学習 | 約1週間 | 4 GPU（A100）使用 |
| 事後学習 | 約1週間 x 3シード | 各シード並列実行可能 |
| Bench2Drive評価 | 約1-2日 x 3シード | 220ルート、最大8並列 |
| Longest6評価 | 約1-2日 x 3シード | 36ルート、最大8並列 |
| Town13評価 | 約1-2日 x 3シード | 20ルート、最大8並列 |

### SLURMなしでの利用

init.shの関数群はSLURMの有無を自動検出し、SLURMがない環境ではローカル実行にフォールバックします。

`train()`関数の判定ロジック:
```bash
if [[ -z "$SLURM_JOB_ID" && $(which sbatch) ]]; then
    # SLURMあり: sbatchでジョブ投入
    sbatch ... slurm/train.sh
else
    # SLURMなし: 直接実行
    bash slurm/train.sh
fi
```

`train.sh`内でもGPU数の自動検出を行います:
```bash
if [ -z "$SLURM_JOB_ID" ]; then
    nproc_per_node=1       # ローカル: GPU 1枚
else
    nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
```

評価パイプラインでも同様に、`evaluate_utils.py`の`is_on_slurm()`が`sbatch`コマンドの存在を確認し、ローカル環境では`bash`で直接スクリプトを実行し、`subprocess.Popen`でプロセスを管理します。

これにより、個人のGPUマシンでも同じスクリプトを使って学習・評価が可能です。

### 障害対応パターン

LEADのSLURMインフラには、複数の障害対応メカニズムが組み込まれています。

1. 学習中のクラッシュ

`resume()`関数でチェックポイントから再開します。復元される状態:
- モデルの重み（model_*.pth）
- optimizer、scheduler、scalerの状態（`continue_epoch=true`で有効）
- WandBのランID（ログの連続性を保持）

```bash
# resume関数は最新のmodel_*.pthを自動検出する
resume outputs/training/001_example/000_pretrain1_0/251018_092144
train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
```

2. 評価ルートの失敗

`SlurmJobPool`がジョブ完了時にJSON結果ファイルを検証します。以下の条件で失敗と判定し、自動再投入します:
- JSONファイルが存在しない
- `score_route`がほぼ0（ソフトウェア障害を示唆）
- ステータスが`Failed`系（Agent crashed, Simulation crashed等）

再試行回数はデータセットごとの設定ファイルで制御されます。

3. CARLAクラッシュ

データ収集パイプラインでは、ログファイルのエラーパターンを20秒ごとに監視します:
- `Actor ... not found!` → scancelで強制終了 → 再投入
- `Watchdog exception - Timeout` → scancelで強制終了 → 再投入
- `Engine crash handling finished; re-raising signal 11` → scancelで強制終了 → 再投入

評価パイプラインでは、各ルートスクリプト内で`clean_carla.sh`を実行してCARLAプロセスをクリーンアップします。

4. SIGINT/SIGTERM（手動中断）

`SlurmJobPool`にシグナルハンドラが登録されています:
- SLURM環境: 全実行中ジョブに`scancel`を発行
- ローカル環境: 全プロセスに`kill -9`を発行 + `clean_carla.sh`でCARLAを終了
- WandBのランを適切に終了

```python
signal.signal(signal.SIGINT, slurm_job_pool.signal_handler)
signal.signal(signal.SIGTERM, slurm_job_pool.signal_handler)
```
