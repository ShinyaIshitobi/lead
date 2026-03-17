# LEAD セットアップ手順 (RTX 5070 Ti / Blackwell GPU)

RTX 5070 Ti (sm_120) で LEAD を動かすための手順。
conda は使わず uv + direnv で環境を管理する。

---

## 前提条件

- Ubuntu 24.04
- NVIDIA GeForce RTX 5070 Ti (または Blackwell 世代 GPU)
- uv インストール済み
- direnv インストール済み (`sudo apt install direnv`)
- zshrc に `eval "$(direnv hook zsh)"` を追加済み

---

## 1. リポジトリのクローン

```bash
git clone https://github.com/autonomousvision/lead.git
cd lead
```

---

## 2. .envrc の作成

プロジェクトルートに `.envrc` を作成する（`.bashrc` への書き込みは不要）。

```bash
cat > .envrc << 'EOF'
# LEAD project environment
export LEAD_PROJECT_ROOT="$(pwd)"

# CARLA
export CARLA_VERSION="0915"
export CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_${CARLA_VERSION}"

# Python paths (CARLA Python API)
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"

# System paths
PATH_add "${LEAD_PROJECT_ROOT}"
PATH_add "${LEAD_PROJECT_ROOT}/scripts"

# NavSim
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"

# Py123D
export PY123D_DATA_ROOT="${LEAD_PROJECT_ROOT}/data/carla_leaderboard2_py123d"

# Activate venv (created by: uv venv --python 3.10)
source .venv/bin/activate
EOF

direnv allow
```

---

## 3. Python 仮想環境の作成と依存パッケージのインストール

```bash
uv venv --python 3.10
uv pip install -r requirements.rtx5070ti.txt --index-strategy unsafe-best-match
uv pip install -e .
```

### requirements.rtx5070ti.txt について

オリジナルの `requirements.txt` は PyTorch 2.5.0 + CUDA 12.4 を使用しており、RTX 5070 Ti (sm_120) に対応していない。
`requirements.rtx5070ti.txt` は以下の変更を加えた環境のフリーズ版:

| パッケージ | オリジナル | RTX 5070 Ti 版 |
|-----------|-----------|---------------|
| torch | 2.5.0+cu124 | 2.7.1+cu128 |
| torchvision | 0.20.0 | 0.22.1+cu128 |
| torchaudio | 2.5.0 | 2.7.1+cu128 |
| numpy | 1.26.0 | 1.26.0 (変更なし) |
| setuptools | (なし) | 80.10.2 (`pkg_resources` のため) |

---

## 4. システム依存パッケージのインストール

```bash
sudo apt install ffmpeg
```

ffmpeg がないと評価時に `RuntimeError: ffmpeg is not installed` で失敗する。

---

## 5. CARLA のセットアップ

```bash
bash scripts/setup_carla.sh
```

約 20GB のダウンロード + 展開で合計 60GB 程度のディスクを使用する。
CDN が遅い場合がある（5-10Mbps 程度）。

---

## 6. GPU 登録の追加

`lead/training/config_training.py` の `gpu_name` プロパティに RTX 5070 Ti を追加する必要がある。

```python
# config_training.py の gpu_name プロパティ内に追加
elif "rtx 5070" in name:
    return "rtx5070ti"
```

`use_mixed_precision_training` プロパティにも追加:

```python
return self.gpu_name in ["a100", "l40s", "rtx5070ti"]
```

---

## 7. チェックポイントのダウンロード

```bash
# テスト用に1つだけダウンロード (ResNet34, 264MB)
bash scripts/download_one_checkpoint.sh
```

`outputs/checkpoints/tfv6_resnet34/` に保存される。

---

## 8. CARLA の起動

### ヘッドレスモード（ウィンドウなし）

```bash
bash scripts/start_carla.sh
```

### ウィンドウ付き（走行を目視確認したい場合）

```bash
$CARLA_ROOT/CarlaUE4.sh \
    -quality-level=Poor \
    -world-port=2000 \
    -resx=1920 \
    -resy=1080 \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=2001 &
```

`-resx` / `-resy` でウィンドウサイズを変更可能。

### CARLA の停止

```bash
kill $(pgrep -f CarlaUE4)
```

### ポート競合時

`Address already in use` エラーが出る場合は、前のプロセスが残っている。

```bash
kill $(pgrep -f CarlaUE4)
sleep 3
ss -tlnp | grep 2000  # ポートが空いたか確認
```

---

## 9. モデル評価の実行

CARLA が起動している状態で、別ターミナルから実行する。

```bash
# Longest6 ベンチマーク (1ルート)
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/longest6/00.xml
```

結果は `outputs/local_evaluation/00/` に保存される:

- `*_demo.mp4` - デモ動画
- `*_debug.mp4` - デバッグ動画
- `infractions.json` - 違反記録
- `metric_info.json` - 評価メトリクス

---

## 10. Bench2Drive ベンチマークの実行

Bench2Drive は 220 本の短距離ルート（各約 150m）で構成されるベンチマーク。
44 種類の交通シナリオ（右折時の歩行者横断、信号変化への対応等）への対処能力を個別に評価できる。

### ルートファイルの準備

リポジトリには `3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml` に全 220 ルートが 1 ファイルにまとまっているが、
評価スクリプトは個別ルートファイル（`data/benchmark_routes/bench2drive/{id}.xml`）を期待する。
以下のスクリプトで分割する:

```bash
python -c "
import xml.etree.ElementTree as ET
import os

tree = ET.parse('3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
root = tree.getroot()
outdir = 'data/benchmark_routes/bench2drive'
os.makedirs(outdir, exist_ok=True)

for route in root.findall('route'):
    rid = route.get('id')
    new_root = ET.Element('routes')
    new_root.append(route)
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree)
    new_tree.write(f'{outdir}/{rid}.xml', xml_declaration=False)

print(f'Split {len(root.findall(\"route\"))} routes')
"
```

### 1 ルートで実行

```bash
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/bench2drive/23687.xml \
  --bench2drive
```

### 別のルートを試す

`data/benchmark_routes/bench2drive/` 内の任意の XML を指定できる:

```bash
# ルート一覧を確認
ls data/benchmark_routes/bench2drive/

# 別のルートで実行
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/bench2drive/1711.xml \
  --bench2drive
```

### bash スクリプトで実行する場合

```bash
bash scripts/eval_bench2drive.sh
```

このスクリプトは Bench2Drive 専用の leaderboard evaluator（`3rd_party/Bench2Drive/leaderboard/`）を使用する。
デフォルトのチェックポイントは `outputs/checkpoints/noradar_resnet34` なので、
`tfv6_resnet34` を使う場合はスクリプト内の `CHECKPOINT_DIR` を書き換えること。

---

## トラブルシューティング

### `No module named 'pkg_resources'`

setuptools 82+ では `pkg_resources` が分離された。80.x 系が必要:

```bash
uv pip install "setuptools<81"
```

### `np.maximum_sctype was removed`

transforms3d が古い。アップグレードで解決:

```bash
uv pip install --upgrade transforms3d
```

### `NVIDIA GeForce RTX 5070 Ti ... is not compatible with the current PyTorch installation`

PyTorch 2.5.0 は sm_120 (Blackwell) に対応していない。2.7.1+cu128 が必要。
`requirements.rtx5070ti.txt` を使用すること。

### `Unknown GPU name: nvidia geforce rtx 5070 ti`

上記「6. GPU 登録の追加」を参照。
