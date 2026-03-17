# lead/inference/ ディレクトリ詳細解説

このドキュメントでは、LEADの推論(inference)パイプライン全体を解説します。
対象読者はEnd-to-End自動運転の初学者です。各モジュールの役割、データの流れ、制御アルゴリズムの仕組みを順を追って説明します。

---

## 学習トピック

このドキュメントを通じて、以下の概念を学ぶことができます。

- CARLAシミュレータとの統合方法（センサー取得からアクチュエータ制御まで）
- Train-Testミスマッチを防ぐための前処理テクニック
- PID制御器の原理と自動運転への適用
- モデルアンサンブルによる推論の安定化
- ヒューリスティックベースのポストプロセッサ（スタック検知、一時停止標識対応）
- Non-Maximum Suppression（NMS）によるバウンディングボックスのフィルタリング
- Closed-Loop（閉ループ）制御とOpen-Loop（開ループ）推論の違い

---

## 全体アーキテクチャの概要

推論パイプラインの処理フローは以下の通りです。

```
CARLAセンサー → 前処理(tick) → テンソル化 → ニューラルネット推論 → 後処理 → PID制御 → VehicleControl
```

主要なファイル構成:

| ファイル | 役割 |
|--------|------|
| sensor_agent.py | CARLAとの統合エントリーポイント |
| closed_loop_inference.py | PID制御器を使ったClosed-Loop推論 |
| open_loop_inference.py | アンサンブル推論とモデルロード |
| config_closed_loop.py | Closed-Loop推論の設定パラメータ |
| inference_utils.py | NMSなどのユーティリティ関数 |
| video_recorder.py | 評価時の動画記録 |

また、共通モジュールとして `lead/common/pid_controller.py` にPID制御器の実装があります。

---

## sensor_agent.py - CARLAとの統合

### クラス構造

`SensorAgent` は2つの基底クラスを継承しています。

- `BaseAgent`: LEADプロジェクト共通の基底クラス
- `autonomous_agent.AutonomousAgent`: CARLAのLeaderboardが要求するインターフェース

CARLAのLeaderboardフレームワークは、エージェントに対して `setup()`, `sensors()`, `tick()`, `run_step()` というメソッドを順番に呼び出します。SensorAgentはこれらを実装することで、CARLAシミュレータ内で自律走行を実現します。

### setup() - 初期化

setup()メソッドでは以下を行います。

1. `ClosedLoopConfig` の読み込み（環境変数からオーバーライド可能）
2. 学習時に保存された `config.json` のロード
3. `ClosedLoopInference` オブジェクトの生成（モデルの読み込みを含む）
4. ポストプロセッサの初期化（ForceMovePostProcessor、StopSignPostProcessor）

### sensors() - センサー構成

`av_sensor_setup` 関数を呼び出し、CARLAに搭載するセンサー群を定義します。
一般的な構成として、カメラ（複数台）、LiDAR、レーダーが含まれます。

### tick() - フレームごとの前処理

tick()はCARLAから受信した生のセンサーデータを、モデルが期待する形式に変換する重要なステップです。
ここで行われる前処理は、学習時のデータ処理と一致させることでtrain-testミスマッチを防ぎます。

処理内容:

- JPEG圧縮シミュレーション: 学習データがJPEGで保存されていたため、推論時にも意図的にJPEG圧縮と展開を行い、圧縮アーティファクトを再現します。品質はデフォルトで90です。
- 水平視野角(Horizontal FOV)の縮小: 学習データと一致するようにカメラ画像の視野角を調整します。
- カメラ選択: 使用するカメラを設定に基づいて選択します。
- Adaptive Pop Distance: ウェイポイントが密集している場合、ルートプランナーのpop distanceを適応的に調整し、遠すぎるターゲットポイントをスキップします。
- LiDAR前処理:
  1. 複数フレームのポイントクラウドを蓄積
  2. 不要な点のフィルタリング
  3. laspy量子化シミュレーション（学習データのフォーマットに合わせる）
  4. ラスタライズ（BEV画像への変換）
  5. 圧縮/展開シミュレーション（学習時のデータパイプラインを再現）
- レーダー前処理

### run_step() - メイン推論ループ

毎フレーム呼び出されるメインの推論関数です。処理の流れは以下の通りです。

1. 前処理済みデータをPyTorchテンソルに変換
2. ニューラルネットワークのforward pass実行
3. 後処理（ポストプロセッサの適用）
4. `carla.VehicleControl` オブジェクトの生成（ステアリング、スロットル、ブレーキ）

加えて以下も行います。

- 違反追跡(Infraction Tracking): 離散的な違反（衝突など）と連続的な違反（車線逸脱など）を区別して記録します。
- 動画記録: demo、debug、input、gridの4種類の動画/画像を記録できます。デバッグや評価結果の確認に使用します。

---

## ForceMovePostProcessor - スタック検知と脱出

自動運転車が動けなくなる（スタックする）状況に対処するためのヒューリスティックです。

### 動作原理

1. 車両の速度が0.1 m/s未満の場合、内部カウンタ(`stuck_detector`)がインクリメントされます
2. カウンタが1100フレーム（約55秒、20FPS想定）に達すると、強制的に前進を開始します
3. 前進時のスロットルは0.4に設定されます
4. 前進は20フレーム（約1秒）継続します

### 安全チェック

強制前進を行う前に、車両前方のボックス領域内のLiDARポイントをサンプリングします。
前方に障害物（ポイントクラウド）が検出された場合、強制前進を中止します。
これにより、前方に車両や障害物がある状態での盲目的な前進を防ぎます。

### なぜ1100フレームか

CARLAの評価では赤信号での停止が必要です。赤信号の待ち時間よりも十分に長い閾値を設定することで、赤信号待ちをスタックと誤判定しないようにしています。

---

## StopSignPostProcessor - 一時停止標識への対応

CenterNetが検出したバウンディングボックスに基づいて、一時停止標識（Stop Sign）に対応するポストプロセッサです。

### 動作フロー

1. CenterNetの出力からStop Signクラスのバウンディングボックスを取得
2. 距離が閾値（1.0m）以内の場合、フルブレーキを適用
3. 一時停止標識を通過した後、120フレームのクールダウン期間を設ける
4. クールダウン後、40フレームの間は減速走行を行う

### デフォルト設定

`slower_for_stop_sign` はデフォルトでFalse（無効）に設定されています。
必要に応じて設定ファイルや環境変数で有効化できます。

---

## closed_loop_inference.py - Closed-Loop推論

### Closed-LoopとOpen-Loopの違い

- Open-Loop推論: モデルの出力（ウェイポイントなど）をそのまま評価する。実際の車両制御は行わない。
- Closed-Loop推論: モデルの出力をPID制御器に入力し、実際にステアリング・スロットル・ブレーキを計算して車両を制御する。制御結果が次のフレームのセンサー入力に影響する（フィードバックループ）。

`ClosedLoopInference` は `OpenLoopInference` を継承し、PID制御器によるClosed-Loop制御を追加しています。

### 4つのPID制御器

ClosedLoopInferenceは4つのPID制御器を初期化します。

1. 横方向ウェイポイント制御器 (lateral_waypoint_controller)
   - 用途: ウェイポイントベースのステアリング制御
   - パラメータ: turn_kp=1.25, turn_ki=0.75, turn_kd=0.3

2. 縦方向ウェイポイント制御器 (longitudinal_waypoint_controller)
   - 用途: ウェイポイントベースの速度制御
   - パラメータ: speed_kp=1.75, speed_ki=1.0, speed_kd=2.0

3. 横方向ルート制御器 (lateral_route_controller)
   - 用途: ルートベースのステアリング制御（速度依存のルックアヘッド距離を使用）
   - パラメータ: lateral_k_p=3.118, lateral_k_d=1.378, lateral_k_i=0.641

4. 縦方向目標速度制御器 (longitudinal_target_speed_controller)
   - 用途: モデルが予測した目標速度に追従する制御

### execute_waypoints() - ウェイポイントベースの制御

1. 現在の速度に基づいてaim distance（照準距離）を決定
2. ウェイポイント上のaim pointに向かう角度を計算
3. 角度誤差を横方向PID制御器に入力 → ステアリング値を取得
4. 目標速度と現在速度の差を縦方向PID制御器に入力 → スロットル/ブレーキ値を取得

### execute_route_and_target_speed() - ルート+目標速度ベースの制御

1. 横方向: LateralPIDControllerを使用（速度依存のルックアヘッド距離で先読み位置を決定し、heading errorを計算）
2. 縦方向: ExpertLongitudinalControllerを使用（PIDではなく線形回帰ベースの制御）

### モダリティ選択

設定により、最終的な制御値をどのPID制御器の出力から取得するかを選択できます。

- steer_modality: "route"（デフォルト）→ ルート制御器のステアリング値を使用
- throttle_modality: "target_speed"（デフォルト）→ 目標速度制御器のスロットル値を使用
- brake_modality: "target_speed"（デフォルト）→ 目標速度制御器のブレーキ値を使用

デフォルト設定では、ステアリングはルートベース、速度制御は目標速度ベースという組み合わせになっています。
これにより、ルートの形状に沿った正確なステアリングと、モデルが予測した適切な速度への追従を両立しています。

---

## open_loop_inference.py - Open-Loop推論とアンサンブル

### モデルのロード

OpenLoopInferenceは複数のチェックポイントをロードし、アンサンブル推論を行います。
複数のモデルの予測を統合することで、単一モデルよりも安定した推論結果を得られます。

### アンサンブル手法

- ensemble_planning_decoder(): 複数モデルのウェイポイント予測を平均化。速度ロジットも平均化した後、softmaxで確率分布に変換し、速度クラスをデコードします。
- ensemble_bounding_boxes(): 全モデルの検出結果を統合し、NMS（Non-Maximum Suppression）で重複を除去します。
- ensemble_bev_semantic(): BEVセマンティックマップのアンサンブル。フリースペース（チャネル0）はmin演算（保守的に安全領域を推定）、障害物はmax演算（見逃しを防ぐ）を使用します。

### ブレーキオーバーライド

速度クラス0（停止）の信頼度が `brake_threshold` を超えた場合、予測速度を強制的に0に設定します。
これは、信号待ちや一時停止などの確実な停止が必要な場面での安全性を高めるための仕組みです。

---

## config_closed_loop.py - 設定パラメータ

`ClosedLoopConfig` は `OpenLoopConfig` を継承し、Closed-Loop推論に必要な全パラメータを定義します。
環境変数 `LEAD_CLOSED_LOOP_CONFIG` からJSONファイルを読み込んでオーバーライドできます。

### 主要パラメータ一覧

画像処理:
- jpeg_quality = 90: JPEG圧縮シミュレーションの品質

制御モダリティ:
- steer_modality = "route": ステアリングの制御元
- throttle_modality = "target_speed": スロットルの制御元
- brake_modality = "target_speed": ブレーキの制御元

カルマンフィルタ:
- use_kalman_filter = False: カルマンフィルタによるGPS平滑化の有無

スタック検知:
- sensor_agent_stuck_threshold = 1100: スタック判定フレーム数
- sensor_agent_stuck_move_duration = 20: 強制前進フレーム数
- sensor_agent_stuck_throttle = 0.4: 強制前進時のスロットル値

一時停止標識:
- slower_for_stop_sign = False: 一時停止標識対応の有効/無効
- slower_for_stop_sign_dist_threshold = 1.0: 停止距離閾値(m)
- slower_for_stop_sign_cool_down = 120: クールダウンフレーム数
- slower_for_stop_sign_count = 40: 減速走行フレーム数

PID制御器（横方向ウェイポイント）:
- turn_kp = 1.25, turn_ki = 0.75, turn_kd = 0.3, turn_n = 20

PID制御器（縦方向ウェイポイント/目標速度）:
- speed_kp = 1.75, speed_ki = 1.0, speed_kd = 2.0, speed_n = 20

PID制御器（横方向ルート）:
- lateral_k_p = 3.118, lateral_k_d = 1.378, lateral_k_i = 0.641, lateral_n = 6

動画出力:
- produce_demo_image / produce_demo_video: デモ用出力
- produce_debug_image / produce_debug_video: デバッグ用出力
- produce_input_image / produce_input_video: 入力データの可視化
- produce_grid_image / produce_grid_video: デモ+入力を縦に結合した出力

---

## PID制御器 (lead/common/pid_controller.py)

### PID制御とは

PID制御は、目標値と現在値の誤差(error)に基づいて制御入力を計算する古典的なフィードバック制御手法です。

- P(比例): 現在の誤差に比例した制御入力。大きいほど応答が速いが、振動しやすい。
- I(積分): 過去の誤差の累積に比例した制御入力。定常偏差を解消するが、過大だとオーバーシュートする。
- D(微分): 誤差の変化率に比例した制御入力。振動を抑制し、安定性を高める。

出力 = Kp * error + Ki * integral(error) + Kd * derivative(error)

### PIDController（汎用クラス）

- スライディングウィンドウ方式でerror履歴を管理します
- 積分項はウィンドウ内のerrorの平均値を使用
- 微分項はウィンドウの最新値と最古値の差分を使用
- ウィンドウサイズ(n)のデフォルトは20

### LateralPIDController（横方向制御）

速度に依存したルックアヘッド距離(lookahead distance)を使用する横方向PID制御器です。

- ルックアヘッド距離の範囲: 2.4m - 10.5m
- 速度が高いほどルックアヘッド距離が長くなり、先の経路を見ながら滑らかにステアリングします
- heading error（車両の向きとルートの向きの差）をPID制御器に入力してステアリング値を計算

### ExpertLateralPIDController

エキスパート（教師あり学習の教師データ生成に使用するコントローラ）向けにチューニングされた横方向PID制御器です。

### ExpertLongitudinalController

PIDではなく、線形回帰を使用した縦方向制御器です。
6つの特徴量（速度、距離、角度など）から直接スロットル/ブレーキ値を予測します。
PIDよりも柔軟な制御が可能で、学習データから最適なパラメータを獲得しています。

---

## inference_utils.py - ユーティリティ関数

### Non-Maximum Suppression (NMS)

物体検出モデルは同じ物体に対して複数の重複するバウンディングボックスを出力することがあります。
NMSはこれらの重複を除去し、各物体に対して最も信頼度の高い1つのバウンディングボックスだけを残すアルゴリズムです。

non_maximum_suppression()の処理フロー:

1. 全モデルのバウンディングボックスを統合
2. 信頼度(confidence)で降順にソート
3. 最も信頼度の高いボックスを選択
4. 選択したボックスとのIoUが閾値を超えるボックスを除去
5. 残りのボックスに対して3-4を繰り返す

### rect_polygon() - 回転矩形の生成

shapely Polygonとして回転した矩形を作成します。
中心座標(x, y)、幅、高さ、回転角度(ラジアン)を指定して使用します。

### iou_bbs() - 回転バウンディングボックスのIoU計算

2つの回転バウンディングボックス間のIoU(Intersection over Union)を計算します。
通常の軸平行バウンディングボックスと異なり、任意の角度に回転したボックス同士のIoUを正確に計算するため、shapely Polygonの幾何学演算を使用しています。

IoU = 交差領域の面積 / 和集合領域の面積

IoUが1に近いほど2つのボックスは重複しており、0に近いほど離れています。

---

## データフローのまとめ

推論全体のデータフローを整理すると以下のようになります。

```
1. CARLAセンサー取得
   カメラ画像、LiDARポイントクラウド、レーダーデータ、GPS/IMU
         |
2. tick() - 前処理
   JPEG圧縮シミュレーション、FOV調整、LiDARラスタライズなど
         |
3. run_step() - テンソル変換
   NumPy配列 → PyTorchテンソル → GPU転送
         |
4. OpenLoopInference - アンサンブル推論
   複数モデルのforward pass → ウェイポイント平均化 → NMS → BEVアンサンブル
         |
5. ClosedLoopInference - PID制御
   ウェイポイント/ルート → PID制御器 → ステアリング/スロットル/ブレーキ
         |
6. ポストプロセッサ
   ForceMovePostProcessor（スタック脱出）
   StopSignPostProcessor（一時停止対応）
         |
7. carla.VehicleControl
   最終的な制御値をCARLAに送信
```

---

## 初学者向けの補足

### なぜtrain-testミスマッチが問題になるのか

ニューラルネットワークは学習データの分布に最適化されます。推論時のデータが学習時と異なる分布を持つ場合、性能が大幅に低下する可能性があります。例えば、学習データがJPEG圧縮された画像で作られていた場合、推論時に非圧縮の画像を入力すると、微妙なピクセル値の違いが蓄積して予測精度が落ちます。tick()内の各種シミュレーション処理はこの問題を防ぐためのものです。

### なぜアンサンブルを使うのか

単一モデルは特定の入力パターンに対して不安定な予測を出すことがあります。複数のモデル（異なるランダムシードや異なるエポックで学習したもの）の予測を統合することで、個々のモデルの弱点を補い合い、全体として安定した推論結果を得ることができます。

### PIDパラメータのチューニング

PID制御器の各ゲイン(Kp, Ki, Kd)は、シミュレーション環境で試行錯誤的にチューニングされたものです。ルート制御器の精密なパラメータ値（例: lateral_k_p=3.118357247806046）は、自動パラメータ最適化の結果と考えられます。パラメータの変更は走行挙動に直接影響するため、慎重に行う必要があります。
