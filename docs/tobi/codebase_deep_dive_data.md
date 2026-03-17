# コードベース詳解: データパイプライン編

このドキュメントでは、LEADプロジェクトのデータ読み込み・管理・前処理に関わる3つのモジュールを解説します。
対象は `lead/data_loader/`、`lead/data_buckets/`、`lead/common/` です。

End-to-End自動運転の初学者が、データがどのように収集・整理・変換されてモデルに入力されるのかを理解できることを目指しています。

---

## 学習トピック一覧

このドキュメントを読むことで、以下のトピックを学ぶことができます。

- PyTorch Datasetクラスを使ったマルチセンサデータの読み込みパターン
- シミュレータ(CARLA)と実世界データ(Waymo, NavSim)の座標系の違いと変換
- LiDAR点群をBird's Eye View(BEV)画像にラスタライズする方法
- CenterNetスタイルのヒートマップラベル生成
- カリキュラム学習のためのバケットベースデータ管理
- センサ摂動(perturbation)によるデータ拡張
- カルマンフィルタによるGPSノイズ低減
- RANSACによる地面点除去
- 効率的なキャッシュ戦略(ディスクキャッシュ、セッションキャッシュ、永続キャッシュ)

---

## 1. data_loader/ -- データ読み込みモジュール

### 1.1 carla_dataset.py -- CARLAデータセット

ファイルパス: `lead/data_loader/carla_dataset.py`

CARLAData クラスは PyTorch の Dataset を継承しており、CARLAシミュレータで収集されたマルチセンサデータを読み込みます。
LEADプロジェクトにおける主要なデータローダです。

#### データ読み込みフロー

`__getitem__` メソッドでは、以下の順序でデータを処理します。

1. メタデータ pickle ファイルの読み込み
2. センサ摂動の適用判定(確率ベースで通常/摂動済みデータを選択)
3. センサデータの読み込み(RGB画像、セマンティック、深度、LiDAR、レーダー)
4. 色空間のデータ拡張(augmentation)
5. LiDAR点群のBEVラスタライズ
6. CenterNetスタイルのラベル生成

#### バケットベースのデータ管理

CARLAData は `config.carla_bucket_collection` を通じてバケットコレクションと連携します。
バケットコレクションがデータのインデックスを管理し、カリキュラム学習のためのサンプリング比率を制御します。

#### キャッシュ戦略

CARLADataは3層のキャッシュ構造を持っています。

- memory_cache: Python辞書によるインメモリキャッシュ(最速だがメモリ消費大)
- training_session_cache: diskcacheライブラリによるセッションキャッシュ(学習セッション間で共有)
- persistent_cache: PersistentCacheクラスによるLZMA圧縮ディスクキャッシュ(永続保存)

#### センサ摂動(Sensor Perturbation)

データ拡張の一種として、カメラの位置と回転にランダムな摂動を加えます。
摂動済みのデータは収集時に別ファイルとして保存されており、学習時に確率的に選択されます。

- 適用確率: `use_sensor_perburtation_prob`(0.5〜0.8程度)
- 平行移動: 0.1m〜1.0mの範囲
- 回転: 5度〜12.5度の範囲

この仕組みにより、カメラの取り付け誤差やドメインシフトに対するロバスト性を獲得します。

#### 色空間データ拡張

imgaugライブラリを使った画像拡張パイプラインが適用されます。

- GaussianBlur: ガウスぼかし
- AdditiveGaussianNoise: ガウスノイズ
- CoarseDropout: ランダムなピクセル領域のドロップアウト
- MultiplyBrightness / GammaContrast: 明度・コントラスト変化

#### 出力データの構造

`__getitem__` が返す辞書には以下のキーが含まれます。

- rgb: RGB画像テンソル
- semantic: セマンティックセグメンテーション
- depth: 深度画像
- lidar: LiDARのBEVラスタライズ画像
- radar: レーダーデータ
- bboxes: バウンディングボックス(位置、サイズ、yaw、速度、クラス）
- target_points: 目標地点座標
- commands: ナビゲーションコマンド
- metadata: 各種メタ情報（町名、シナリオタイプ、天候、速度など）

---

### 1.2 navsim_dataset.py -- NavSimデータセット

ファイルパス: `lead/data_loader/navsim_dataset.py`

NavsimData クラスは NavSim データセットを読み込みます。
NavSimはnuPlanベースのクローズドループシミュレーション環境であり、gzip圧縮された特徴量/ターゲットファイルのペアとしてデータが保存されています。

#### 座標系の変換

NavSimとCARLAでは Y軸の向きが逆です。NavSimから読み込んだデータは、CARLAの座標系に合わせるためにY座標を反転(符号を反転)させます。
ヘディング角も同様にCARLAの規約に揃えられます。

#### BEVセマンティックマップの変換

NavSimのBEVセマンティックマップはCARLAの向きと90度ずれているため、読み込み時に90度回転を行います。

#### アップサンプリング/ダウンサンプリング

設定ファイルの `navsim_num_samples` に基づいて、データ数の調整が行われます。

- 元のデータ数より多い場合: 全データを繰り返し使い、余りはランダムにサンプリング
- 元のデータ数より少ない場合: ランダムにダウンサンプリング

エポックごとに異なるシードでシャッフルされ、再現性を保ちながら多様なサンプリングが実現されます。

---

### 1.3 waymo_e2e_dataset.py -- Waymoデータセット

ファイルパス: `lead/data_loader/waymo_e2e_dataset.py`

WODE2EData クラスは Waymo Open Dataset E2E Challengeのデータを読み込みます。
JPEG画像とJSON.GZ形式のメタデータで構成されています。

#### データ分割

コンストラクタの `training`、`val`、`test` フラグにより、3つのデータ分割のいずれかを選択します。
これらのフラグは排他的であり、必ず1つだけ True にする必要があります。

#### RFS(Repeated Factor Sampling)

バリデーション時に compute_rfs を利用し、希少なサンプルに対するサンプリング頻度を調整します。
これにより、出現頻度の低いシナリオの評価精度が向上します。

#### サンプリング制御

学習時は `waymo_e2e_num_training_samples` パラメータにより、NavSimと同様のアップ/ダウンサンプリングが行われます。

---

### 1.4 training_cache.py -- キャッシュシステム

ファイルパス: `lead/data_loader/training_cache.py`

学習データの読み込みを高速化するためのキャッシュ機構を提供します。

#### SensorData データクラス

非圧縮のセンサデータを保持するデータクラスです。以下のフィールドを持ちます。

- image: RGB画像 (uint8, H x W x 3)
- rasterized_lidar: LiDAR BEV画像 (float32, H x W)
- semantic: セマンティックセグメンテーション (uint8, H x W)
- hdmap: BEVセマンティックマップ (uint8, H x W)
- depth: 深度画像 (float32, H x W)
- boxes: バウンディングボックス (float32, N x features)
- bev_occupancy: BEV占有率マップ (uint8, H x W)
- radars: 4つのレーダーセンサからの点群データのタプル

jaxtyping と beartype による型アノテーションが適用されており、実行時に型チェックが行われます。

#### CompressedSensorData データクラス

SensorData を圧縮した形式です。ディスクへの保存に使用されます。

- RGB画像: JPEG圧縮(品質パラメータで制御)
- LiDAR BEV: float値を16bit整数にスケーリング後、PNGロスレス圧縮
- セマンティック/HDマップ/BEV占有率: PNGロスレス圧縮
- 深度: 8bitエンコード後、PNGロスレス圧縮
- レーダー: numpy npz圧縮(float16精度)
- バウンディングボックス: 非圧縮（データサイズが小さいため）

#### PersistentCache クラス

ディスクベースの永続キャッシュです。
キャッシュキー（CacheKey）は、シナリオ名、ルート番号、フレーム番号、摂動有無の組み合わせで構成されます。

- 保存形式: LZMA圧縮された pickle ファイル
- 存在確認のキャッシュ: ファイルシステムへの問い合わせを最小化するため、存在確認結果もメモリにキャッシュ
- パス構造: `{carla_root}/cache/{scenario}/{route}/{cache_path}/{normal|perturbated}/{frame}.pkl`

---

### 1.5 carla_dataset_utils.py -- ユーティリティ関数群

ファイルパス: `lead/data_loader/carla_dataset_utils.py`

1000行を超えるユーティリティモジュールで、データの前処理とラベル生成を担当します。

#### rasterize_lidar() -- LiDAR点群のBEVラスタライズ

3D LiDAR点群を2D Bird's Eye View(鳥瞰図)画像に変換する関数です。

処理の流れ:

1. 点群のXY座標をBEV画像のピクセル座標に変換
2. 画像範囲外の点を除去
3. ヒストグラムビニングにより各ピクセルに点の密度を計算
4. 正規化して0〜1の値に変換

この処理により、3次元の点群データが2次元の画像として扱えるようになり、画像ベースのニューラルネットワークへの入力として利用できます。

#### image_augmenter() -- 画像拡張パイプライン

imgaugライブラリを使った確率的な画像拡張パイプラインを生成します。
学習データの多様性を高め、過学習を防ぎます。

#### get_centernet_labels() -- CenterNetラベル生成

CenterNetスタイルの検出ラベルを生成します。各ターゲットについて以下を計算します。

- ヒートマップ: オブジェクト中心にガウシアンカーネルを配置
- wh: バウンディングボックスの幅と高さ
- offset: ピクセル内のサブピクセルオフセット
- yaw: オブジェクトの向き
- velocity: オブジェクトの速度

CenterNetは、アンカーフリーの物体検出手法であり、各オブジェクトの中心点を検出することで物体を認識します。
ヒートマップ上のピーク位置がオブジェクト中心に対応し、そこから各属性を回帰します。

#### perturbate_route() / perturbate_target_point() -- 経路摂動

ルートポイントやターゲットポイントに対して2Dの剛体変換（回転＋平行移動）を適用します。
センサ摂動と組み合わせることで、カメラの位置ずれに対応した教師データを生成します。

---

## 2. data_buckets/ -- カリキュラム学習モジュール

ファイルパス: `lead/data_buckets/`

バケットベースのカリキュラム学習を実現するモジュールです。
学習データをシナリオの種類や走行特性に基づいて分類し、各バケットのサンプリング比率を調整することで、モデルが苦手とするシナリオに重点的に学習させることができます。

### 2.1 カリキュラム学習とは

通常の学習では、データセット内のサンプルが均一にサンプリングされます。
しかし、自動運転データでは「直進走行」のような単純なシナリオが大半を占め、「対向車の信号無視」のような重要だが稀なシナリオは少数です。

カリキュラム学習では、このようなデータの不均衡を解消するために:
- 重要なシナリオのサンプリング比率を高くする
- 単純なシナリオのサンプリング比率を低くする

これにより、限られた学習時間内でより効果的にモデルを訓練できます。

### 2.2 abstract_bucket_collection.py -- 抽象基底クラス

ファイルパス: `lead/data_buckets/abstract_bucket_collection.py`

AbstractBucketCollection はすべてのバケットコレクションの基底クラスです。

主な機能:

- キャッシュの読み込み/保存: バケットの構築は時間がかかるため、一度構築したらLZMA圧縮してディスクに保存
- ルートの走査: データディレクトリ内のルートを順に走査して各バケットに振り分け
- 統計情報の管理: total_routes、trainable_routes、all_frames、trainable_frames を追跡

バケットコレクションが初めて構築される場合は `_build_buckets()` が呼ばれ、2回目以降はキャッシュから読み込まれます。
`force_rebuild_bucket` オプションでキャッシュを無視して再構築することもできます。

### 2.3 bucket.py -- バケットコンテナ

ファイルパス: `lead/data_buckets/bucket.py`

Bucket クラスは、各バケットに属するフレームのファイルパスを管理します。
1フレーム分のデータとして以下のパスを保持します:

- rgb: `{route_dir}/rgb/{seq:04d}.jpg`
- rgb_perturbated: 摂動済みRGB画像
- semantics / semantics_perturbated: セマンティックセグメンテーション
- hdmap / hdmap_perturbated: HDマップ
- depth / depth_perturbated: 深度画像
- lidar: `{route_dir}/lidar/{seq:04d}.laz` (LAZ圧縮LiDAR)
- radar / radar_perturbated: レーダーデータ (npz形式)
- bboxes: バウンディングボックス (pickle形式)
- metas: メタデータ (pickle形式)

`finalize()` メソッドで、全リストを numpy の string_ 配列に変換します。
これは PyTorch DataLoader のマルチプロセス環境でメモリリークを防ぐための対策です。

### 2.4 full_pretrain_bucket_collection.py -- 事前学習用バケット

ファイルパス: `lead/data_buckets/full_pretrain_bucket_collection.py`

FullPretrainBucketCollection は最もシンプルなバケットコレクションです。
単一のバケットにすべてのデータを格納し、フィルタリングは行いません。

事前学習段階では、データの偏りを気にせず全データを均一に使うことが有効な場合があります。

### 2.5 waymo_bucket_collection.py -- Waymo用バケット

ファイルパス: `lead/data_buckets/waymo_bucket_collection.py`

WaymoBucketCollection は Waymo Sim2Real 転移学習のためのバケットコレクションです。
WaymoSim2RealBuckets 列挙型で定義された75個のバケットにデータを分類します。

バケットは大きく2つのカテゴリに分かれます:

シナリオ固有のバケット(約40種類):
- ACCIDENT_SCENARIO: 事故シナリオ
- CONSTRUCTION_OBSTACLE_SCENARIO: 工事障害物シナリオ
- DYNAMIC_OBJECT_CROSSING_SCENARIO: 動的オブジェクト横断
- PEDESTRIAN_CROSSING_SCENARIO: 歩行者横断
- SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO: 信号交差点左折
- HIGHWAY_CUT_IN_SCENARIO: 高速道路割り込み
- VEHICLE_OPENS_DOOR_TWO_WAYS_SCENARIO: 車両ドア開放
- など

一般的なマイニングバケット(約35種類):
- HIGH_ACCELERATION / MEDIUM_ACCELERATION / LOW_ACCELERATION: 加速度レベル別
- HIGH_ROUTE_CURVATURE / MEDIUM_ROUTE_CURVATURE: 経路の曲率別
- RED_TRAFFIC_LIGHT / RED_OVERHEAD_TRAFFIC_LIGHT: 赤信号関連
- STOP_SIGN_HAZARD: 一時停止標識
- VEHICLE_HAZARD: 車両危険
- ENTERING_JUNCTION / CLOSE_TO_JUNCTION: 交差点接近
- OTHERS: 上記のいずれにも該当しないフレーム

#### サンプリング比率の例

`buckets_mixture_per_epoch()` メソッドで各バケットのサンプリング比率を定義します。
一部の例を示します:

- OPPOSITE_VEHICLE_RUNNING_RED_LIGHT_SCENARIO: 5.0 (高優先度)
- PEDESTRIAN_CROSSING_SCENARIO: 5.0 (高優先度)
- PARKING_CUT_IN_SCENARIO: 5.0 (高優先度)
- NON_SIGNALIZED_JUNCTION_LEFT_TURN_SCENARIO: 3.0
- STOP_SIGN_HAZARD: 3.0
- OTHERS: 0.1 (大幅にダウンサンプリング)
- ENTER_ACTOR_FLOW_SCENARIO: 0.0 (使用しない)

比率が1.0を超えるバケットはオーバーサンプリングされ、1.0未満のバケットはアンダーサンプリングされます。

### 2.6 navsim_bucket_collection.py -- NavSim用バケット

ファイルパス: `lead/data_buckets/navsim_bucket_collection.py`

NavSimBucketCollection は WaymoBucketCollection と同様に75個のバケットを持ちます。
NavSim固有の座標系変換を考慮しつつ、同じシナリオ分類ロジックを適用します。

### 2.7 route_filtering.py -- ルートフィルタリング

ファイルパス: `lead/data_buckets/route_filtering.py`

データ収集時にエキスパートドライバーが完璧に走行できなかったルートを除外するためのフィルタリング関数群です。

- route_failed(): ルートが失敗したかチェック。results.json のスコアが100点未満で重大な違反がある場合に True を返す。ただし最低速度違反(min_speed_infractions)のみの場合は許容する
- route_not_finished(): results.json が存在しない（収集が完了していない）場合に True を返す
- route_completed_but_fail(): ルートは完走したがスコアが不完全な場合に True を返す。失敗学習用バケットで使用される

フィルタリングにより、高品質な教師データのみが学習に使用されます。
エキスパートが事故を起こしたルートのデータで学習すると、モデルが誤った行動を学習してしまうためです。

---

## 3. common/ -- 共通ユーティリティモジュール

### 3.1 config_base.py -- 基本設定クラス

ファイルパス: `lead/common/config_base.py`

BaseConfig はデータ収集とモデル学習の両方で使用される設定の基底クラスです。

#### 運動学的自転車モデル(Kinematic Bicycle Model)のパラメータ

自動運転における車両の動きをシミュレートするための数学モデルです。

- time_step = 0.05秒 (20 FPS)
- front_wheel_base = -0.090769015: 前輪軸の位置
- rear_wheel_base = 1.4178275: 後輪軸の位置
- steering_gain = 0.36848336: ステアリング角度→ホイール角度の変換係数
- brake_acceleration = -4.952399 m/s^2: ブレーキ時の減速度
- throttle_acceleration = 0.5633837 m/s^2: アクセル時の加速度

これらのパラメータは World on Rails プロジェクトでチューニングされた値です。

#### カメラ摂動パラメータ

- camera_translation_perturbation_min = 0.1m
- camera_translation_perturbation_max = 1.0m
- camera_rotation_perturbation_min = 5.0度
- camera_rotation_perturbation_max = 12.5度
- camera_rotation_epsilon = 0.5度 (これ以下の回転は無視)

#### LiDAR設定

- point_precision_x/y/z = 0.1: LiDAR点の保存精度(メートル単位)
- max_height_lidar = 10.0m: この高さ以上の点は破棄
- min_height_lidar = -4.0m: この高さ以下の点は破棄
- lidar_accumulation: 複数フレームにわたるLiDAR点群の蓄積を有効にする

#### カメラキャリブレーション

データセットごとに異なるカメラ構成を定義します。

- CARLA_LEADERBOARD2_6CAMERAS: 6台のカメラ(前方3台 + 後方3台)、各カメラ FOV 60度
- CARLA_LEADERBOARD2_3CAMERAS: 3台のカメラ(前方のみ)
- CARLA_LEADERBOARD2_1CAMERA: 1台のカメラ
- NAVSIM_4CAMERAS: 4台のカメラ
- WAYMO_E2E_2025_3CAMERAS: 3台のカメラ

#### overridable_property デコレータ

環境変数による設定値の上書きを可能にするデコレータです。
クラスタ環境などでコードを変更せずにパラメータを調整する際に便利です。

---

### 3.2 constants.py -- 列挙型と定数

ファイルパス: `lead/common/constants.py`

プロジェクト全体で使用される定数と列挙型を定義します。

#### TargetDataset 列挙型

モデルがターゲットとするデータセットの種類です。

- CARLA_LEADERBOARD2_3CAMERAS / 6CAMERAS / 1CAMERA: CARLAシミュレータのバリエーション
- NAVSIM_4CAMERAS: NavSimデータセット
- WAYMO_E2E_2025_3CAMERAS: Waymo E2Eチャレンジ
- CARLA_PY123D_1CAMERA: Py123Dによるカメラ1台構成

#### SourceDataset 列挙型

データの出所を表します。混合データ学習時に、各サンプルがどのデータセットから来たかを追跡します。

- CARLA: CARLAシミュレータ
- NAVSIM: NavSimシミュレータ
- WAYMO_E2E_2025: Waymo実世界データ

#### TransfuserBEVSemanticClass 列挙型 (13クラス)

BEVセマンティックマップの各ピクセルが表すクラスです。

0. UNLABELED: ラベルなし
1. ROAD: 道路
2. LANE_MARKERS: 車線マーカー
3. STOP_SIGNS: 一時停止標識
4. VEHICLE: 車両
5. WALKER: 歩行者
6. OBSTACLE: 障害物
7. PARKING_VEHICLE: 駐車車両
8. SPECIAL_VEHICLE: 特殊車両
9. BIKER: 自転車/バイク乗り
10. TRAFFIC_GREEN: 青信号
11. TRAFFIC_RED_NORMAL: 赤信号（通常）
12. TRAFFIC_RED_NOT_NORMAL: 赤信号（特殊）

#### TransfuserBoundingBoxClass 列挙型 (8クラス)

検出対象のバウンディングボックスクラスです。

0. VEHICLE: 車両
1. WALKER: 歩行者
2. TRAFFIC_LIGHT: 信号機
3. STOP_SIGN: 一時停止標識
4. SPECIAL: 特殊オブジェクト
5. OBSTACLE: 障害物
6. PARKING: 駐車
7. BIKER: 自転車/バイク乗り

#### CarlaSemanticSegmentationClass 列挙型 (31クラス)

CARLAのセマンティックセグメンテーションカメラが出力するクラスです。
主要なものを挙げます:

- Roads(1), SideWalks(2): 道路と歩道
- Building(3), Wall(4), Fence(5): 建物系
- TrafficLight(7), TrafficSign(8): 交通標識系
- Car(14), Truck(15), Bus(16): 車両系
- Pedestrian(12), Rider(13): 人物系

#### CarlaNavigationCommand 列挙型 (7コマンド)

エージェントに与えられるナビゲーションコマンドです。
ルートプランナーがこれらのコマンドを生成し、モデルへの条件付け入力として使用されます。

0. UNKNOWN: 不明
1. LEFT: 左折
2. RIGHT: 右折
3. STRAIGHT: 直進
4. LANEFOLLOW: 車線追従
5. CHANGELANELEFT: 左車線変更
6. CHANGELANERIGHT: 右車線変更

#### SCENARIO_TYPES

CARLAで定義された48種類のシナリオ名の一覧です。
バケットコレクションでのシナリオ分類に使用されます。

---

### 3.3 common_utils.py -- ユーティリティ関数群

ファイルパス: `lead/common/common_utils.py`

842行にわたる汎用ユーティリティモジュールです。

#### 座標変換

自動運転では複数の座標系を扱う必要があります。

- GPS → CARLA座標: GPSの緯度経度をCARLAのワールド座標(メートル)に変換
- LiDAR → 自車座標: LiDAR点群を自車両中心の座標系に変換
- レーダー → 自車座標: レーダーデータを自車両座標系に変換
- conversion_2d / inverse_conversion_2d: 2次元座標の正変換と逆変換

#### 深度画像のエンコード/デコード

深度情報を効率的に保存するためのエンコーディングです。

- encode_depth_8bit: 32bitの深度値を8bitに量子化して保存
- encode_depth_16bit: 32bitの深度値を16bitに量子化して保存
- decode_depth: エンコードされた深度値を元のfloat32に復元

8bitエンコーディングでは情報の損失がありますが、ファイルサイズを大幅に削減できます。

#### 軌跡評価メトリクス

予測軌跡と真値軌跡の差異を評価するための指標です。

- ADE(Average Displacement Error): 各タイムステップにおける予測位置と真値位置のユークリッド距離の平均
- FDE(Final Displacement Error): 最終タイムステップにおける予測位置と真値位置のユークリッド距離
- curvature: 軌跡の曲率計算

#### カメラ視錐台チェック

3D空間の点やバウンディングボックスがカメラの視野内に入っているかを判定します。

- is_point_in_camera_frustum: 1点がカメラの視錐台内にあるかチェック
- is_box_in_camera_frustum: バウンディングボックスの頂点がカメラの視錐台内にあるかチェック

これらの関数は、カメラの内部パラメータ(焦点距離、画像サイズ)と外部パラメータ(位置、姿勢)を使って射影計算を行います。

---

### 3.4 base_agent.py -- エージェント基底クラス

ファイルパス: `lead/common/base_agent.py`

BaseAgent はエキスパートドライバーと学習済みモデルの両方が継承する基底クラスです。
センサデータの前処理とフィルタリングを担当します。

#### 初期化 (setup)

- KalmanFilter: UKF(Unscented Kalman Filter)によるGPSノイズ低減
- RoutePlanner: 目標地点までの経路計画
- センサキュー: LiDAR、レーダーのフレーム蓄積用deque
- Numbaキャッシュのウォームアップ: RANSACの初回コンパイルを事前に実行

#### tick() メソッド -- フレームごとの処理

各シミュレーションステップで呼ばれるメソッドです。

1. GPSフィルタリング: カルマンフィルタを通してノイズを低減
2. 状態計算: 現在位置、速度、ヨー角を計算
3. LiDAR蓄積: 複数フレームのLiDAR点群を自車座標系に変換して蓄積
4. 地面除去: RANSAC による地面点の除去
5. レーダー処理: 自車による反射（エゴ重複）の除去
6. カメラ合成: 複数カメラの画像を水平方向に結合

#### 時系列データバッファリング

過去60フレーム分のデータ（位置、速度、ヨー角）をdequeに保存し、時系列的な文脈情報として利用します。

---

### 3.5 kalman_filter.py -- Unscented Kalman Filter

ファイルパス: `lead/common/kalman_filter.py`

GPSセンサのノイズを低減するためのUnscented Kalman Filter(UKF)の実装です。

#### なぜカルマンフィルタが必要か

CARLAシミュレータのGPSセンサには意図的にノイズが付加されています。
実世界のGPSも同様にノイズがあるため、ノイズに対処する仕組みが必要です。

#### 状態ベクトル

状態は4次元です: [x, y, yaw, speed]

- x, y: CARLA座標系での位置(メートル)
- yaw: 車両の向き(ラジアン)
- speed: 速度(m/s)

#### 遷移モデル

Leaderboard 1.0の運動学的自転車モデルを使用します。
ステアリング、スロットル、ブレーキのコマンドから次の状態を予測します。

#### ノイズパラメータ

- P(状態共分散): 位置に大きめの不確実性(0.5)、角度・速度に小さい不確実性
- R(観測ノイズ): 位置に大きめ(0.5)、角度・速度はほぼノイズなし
- Q(モデルノイズ): 全体的に小さい値(モデルを信頼)

#### RTS Smoothing

フィルタリングに加えて、RTS(Rauch-Tung-Striebel)スムーザーを使った時系列的な平滑化が可能です。
これは、将来のデータも考慮してより正確な状態推定を行う手法で、データ収集時のオフライン処理に使用されます。

filterpy ライブラリの UKF 実装をベースに、制御入力(ステアリング、スロットル、ブレーキ)を考慮するようにカスタマイズされています。

---

### 3.6 ransac.py -- RANSAC地面除去

ファイルパス: `lead/common/ransac.py`

LiDAR点群から地面に属する点を除去するためのモジュールです。
Numba の @njit デコレータにより、JITコンパイルされた高速なネイティブコードで実行されます。

#### なぜ地面除去が必要か

LiDAR点群の大部分は地面からの反射です。
地面の点を含んだままBEV画像を生成すると、道路面が密に埋まってしまい、車両や歩行者などの重要なオブジェクトの検出が困難になります。

#### アルゴリズムの概要

1. 中心点のグリッド生成: BEV範囲内に等間隔で中心点を配置(デフォルト解像度 28m)
2. 放射状セグメント分割: 各中心点の周囲を角度方向にセグメント分割(デフォルト 8セグメント)
3. 初期シード点の選出: 各セグメント内で最も低い点群(N_LPR=64個)のメディアン高さ付近の点を初期シード点として選出
4. 平面推定: 最小二乗法で地面平面を推定(ax + by + d = z)
5. 反復改善: 推定平面に近い点(Th_dist=0.15m以内)を新たなシードとして平面を再推定(N_iter=2回)
6. 地面マスク生成: 最終的に地面と判定された点のブールマスクを返す

各中心点ごとに半径 min_r(1m) 〜 max_r(32m) の範囲内の点のみを処理します。

#### 並列処理

fit_ground_parallel 関数は `@njit(parallel=True)` と `prange` を使って複数の中心点を並列処理します。
BaseAgent の setup 時にダミーデータで一度呼び出すことで、Numba のJITコンパイルキャッシュを事前に生成しています。

---

## 全体アーキテクチャの概要

以下に、データパイプラインの全体的な流れをまとめます。

```
データ収集(CARLAシミュレータ)
    |
    v
生データ保存
(RGB JPEG, LiDAR LAZ, メタデータ pickle, セマンティック PNG, 深度 PNG, レーダー NPZ)
    |
    v
ルートフィルタリング (route_filtering.py)
  - 失敗したルートを除外
  - エキスパートスコアが不完全なルートを除外
    |
    v
バケット構築 (data_buckets/)
  - シナリオ/走行特性に基づいて各フレームをバケットに分類
  - サンプリング比率を設定
  - キャッシュとして保存
    |
    v
データセットクラス (data_loader/)
  - CARLAData / NavsimData / WODE2EData
  - バケットからのインデックスでサンプリング
    |
    v
前処理パイプライン
  - メタデータ読み込み
  - センサ摂動の適用判定
  - センサデータ読み込み (キャッシュ活用)
  - 色拡張
  - LiDAR BEVラスタライズ (地面除去含む)
  - CenterNetラベル生成
    |
    v
PyTorch DataLoader
  - バッチ化
  - モデルへの入力
```

---

## 学習のヒント

このコードベースを理解する上で、以下の順序で学習することを推奨します。

1. まず `common/constants.py` を読んで、プロジェクトで使われる列挙型と定数を把握する
2. `common/config_base.py` でセンサ構成やパラメータの全体像を掴む
3. `data_buckets/bucket.py` と `data_buckets/abstract_bucket_collection.py` でデータの整理方法を理解する
4. `data_loader/carla_dataset.py` の `__getitem__` メソッドを追い、データ読み込みの流れを把握する
5. `data_loader/carla_dataset_utils.py` の各ユーティリティ関数を個別に理解する
6. `common/base_agent.py` の `tick()` メソッドで、推論時のセンサ処理を理解する
7. `common/kalman_filter.py` と `common/ransac.py` でセンサノイズ処理の詳細を学ぶ

各ファイルにはjaxtyping/beartypeによる型アノテーションが付いているため、関数のシグネチャを読むだけでも入出力のテンソル形状を把握できます。
