# LEAD コードベース完全解析

このドキュメント群は、LEADリポジトリの全Pythonファイルを1行ずつ読み解き、クラス・メソッド・依存関係・データフローを網羅的に記録したものである。

## ドキュメント一覧

| ファイル | 内容 | 対象ディレクトリ |
|---------|------|----------------|
| [モデルアーキテクチャ編](codebase_deep_dive_model.md) | TFv6の全コンポーネント | `lead/tfv6/` |
| [学習パイプライン編](codebase_deep_dive_training.md) | 学習ループ、設定、損失関数 | `lead/training/` |
| [エキスパートドライバー編](codebase_deep_dive_expert.md) | LEAD Expert、データ収集 | `lead/expert/` |
| [推論パイプライン編](codebase_deep_dive_inference.md) | 閉ループ推論、PID制御 | `lead/inference/` |
| [データパイプライン編](codebase_deep_dive_data.md) | データローダ、キャッシュ、バケット | `lead/data_loader/`, `lead/data_buckets/`, `lead/common/` |
| [スクリプト・ツール編](codebase_deep_dive_scripts.md) | シェルスクリプト、SLURM、ユーティリティ | `scripts/`, `slurm/`, `notebooks/` |

## 全体アーキテクチャ

```
データ収集フェーズ:
  CARLA Simulator
    → Expert Agent (lead/expert/)
      → センサデータ記録 (RGB, LiDAR, Radar, Semantic, Depth, BBox, HDMap)
      → 摂動データも同時記録
    → 保存先: data/carla_leaderboard2/data/

学習フェーズ:
  データローダ (lead/data_loader/)
    → CARLAData / NavsimData / WODE2EData
    → BucketCollection でカリキュラム学習
    → TrainingCache で高速化
      ↓
  学習ループ (lead/training/train.py)
    → TFv6モデル (lead/tfv6/)
      → TransfuserBackbone: Image + LiDAR の4段階Transformer融合
      → PlanningDecoder: Route + Waypoint + Speed 予測
      → BEVDecoder / PerspectiveDecoder / CenterNetDecoder: 補助タスク
      → RadarDetector: レーダー物体検出
    → 損失計算 → 逆伝播 → パラメータ更新
    → チェックポイント保存

推論フェーズ:
  CARLA Simulator
    → SensorAgent (lead/inference/sensor_agent.py)
      → センサ前処理 (JPEG圧縮シミュレーション, LiDARラスタ化)
      → ClosedLoopInference (lead/inference/closed_loop_inference.py)
        → モデルアンサンブル (複数チェックポイントの平均)
        → PID制御器 (Route/Waypoint/Speed の3系統)
      → 後処理 (停止標識、スタック検出)
    → carla.VehicleControl 出力
```

## モジュール間依存関係

```
lead/common/
  ├── config_base.py      ← 全モジュールの基盤設定
  ├── constants.py         ← 全モジュールが参照するEnum/定数
  ├── common_utils.py      ← 座標変換、深度エンコーディング等
  ├── pid_controller.py    ← inference/ と expert/ が使用
  ├── base_agent.py        ← inference/ と expert/ の共通基底
  ├── kalman_filter.py     ← base_agent.py が使用
  ├── sensor_setup.py      ← inference/ と expert/ が使用
  ├── route_planner.py     ← base_agent.py が使用
  └── ransac.py            ← base_agent.py がLiDAR地面除去に使用

lead/tfv6/
  ├── tfv6.py              ← training/ と inference/ が使用
  ├── transfuser_backbone.py ← tfv6.py が使用
  ├── planning_decoder.py  ← tfv6.py が使用
  ├── bev_decoder.py       ← tfv6.py が使用
  ├── perspective_decoder.py ← tfv6.py が使用
  ├── center_net_decoder.py ← tfv6.py が使用
  ├── radar_detector.py    ← tfv6.py が使用
  └── transfuser_utils.py  ← 上記全てが使用

lead/training/
  ├── train.py             ← tfv6/, data_loader/, data_buckets/ を使用
  ├── config_training.py   ← config_base.py を継承
  └── mixed_training_utils.py ← train.py が使用

lead/data_loader/
  ├── carla_dataset.py     ← training/ が使用
  ├── navsim_dataset.py    ← training/ が使用
  ├── waymo_e2e_dataset.py ← training/ が使用
  └── training_cache.py    ← carla_dataset.py が使用

lead/inference/
  ├── sensor_agent.py      ← leaderboard_wrapper.py が使用
  ├── closed_loop_inference.py ← sensor_agent.py が使用
  ├── open_loop_inference.py ← closed_loop_inference.py が使用
  └── config_closed_loop.py ← config_base.py を継承

lead/expert/
  ├── expert.py            ← leaderboard_wrapper.py が使用
  ├── expert_data.py       ← expert.py が継承
  ├── expert_base.py       ← expert_data.py が継承
  ├── privileged_route_planner.py ← expert.py が使用
  └── hdmap/               ← expert_data.py が使用
```
