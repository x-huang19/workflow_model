# Beam Track Analyzer (Windows EXE)

基于本地 Hugging Face 视觉语言模型（支持 Qwen2.5-VL 与 Qwen3.5-35B-A3B）分析多频带波束历程图，提取轨迹并筛选“仅在 2-3 个频带内出现”的目标轨迹。

## 功能

- 支持本地模型目录加载（`from_pretrained` + `safetensors`）。
- 支持一次输入多张图片，每张图片绑定一个 `band_id`。
- Prompt 与图片列表均由配置文件驱动，无需改代码。
- 程序内执行跨频带轨迹匹配与 2-3 频带筛选。
- 输出 JSON/CSV 结果与运行日志。

## 快速开始（源码方式）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
cp config/runtime.linux.example.yaml config/runtime.local.yaml
# 修改 runtime.local.yaml 中的 model.local_model_dir 和 input.images 路径
beam-track-analyzer --config config/runtime.local.yaml --dry-run
```

正式运行：

```bash
beam-track-analyzer --config config/runtime.local.yaml
```

如果不想 `pip install -e .`，可使用：

```bash
PYTHONPATH=src python3 -m app.cli --config config/runtime.local.yaml
```

## CUDA 11.8 环境（与你提供的版本一致）

如果你要复现你之前可用的版本组合：
- `pytorch-cuda 11.8`
- `torch 2.7.1+cu118`
- `transformers 4.57.7`

可选方案 1（pip）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev-cu118.txt
pip install -e .
```

如果你当前机器就是 CUDA 11.8，建议直接使用上面的 `requirements-dev-cu118.txt`，不要安装通用 `requirements-dev.txt`。

可选方案 2（conda）：

```bash
conda env create -f environment-cu118.yml
conda activate beam-track-cu118
```

## 运行配置

复制并修改配置（Linux 建议从 `config/runtime.linux.example.yaml` 开始）：

- `model.local_model_dir`：本地模型目录（含 safetensors / config / tokenizer 等）
- `model.device_map`：大模型可设为 `auto` 让 Hugging Face 自动分配
- `model.attn_implementation`：推荐 V100 用 `sdpa`
- `input.images`：图片路径与 `band_id`
- `prompt.template_file` 与 `prompt.extra_instruction`
- `matching`：轨迹匹配阈值与频带数量筛选

### 单卡推理示例（指定 GPU）

示例 1：在配置中固定到 `cuda:0`（推荐）

```yaml
model:
  local_model_dir: "D:/models/Qwen3.5-35B-A3B"
  trust_remote_code: true
  dtype: "float16"
  device: "cuda:0"
  device_map: "cuda:0"
  attn_implementation: "sdpa"
  max_new_tokens: 512
  temperature: 0.1
```

示例 2：用环境变量指定物理卡，再在配置里用 `cuda:0`

Windows PowerShell:
```powershell
$env:CUDA_VISIBLE_DEVICES="1"
beam-track-analyzer.exe --config .\\config\\runtime.yaml
```

## 输出文件

- `model_raw_output.json`：模型原始文本输出
- `all_tracks_by_band.json`：每频带轨迹（可关闭中间结果）
- `cross_band_tracks_2_to_3.json`
- `cross_band_tracks_2_to_3.csv`
- `run.log`

## Windows EXE 打包

```bash
pyinstaller --noconfirm --clean build/pyinstaller.spec
```

## GitHub Actions

工作流文件：`.github/workflows/build-windows-exe.yml`

- 触发：
  - `push tag`（`v*`）
  - `workflow_dispatch`
- 流程：安装依赖 -> 运行测试 -> 打包 onedir exe -> 生成 zip -> 上传 artifact
- 当触发为 tag 时，自动发布到 GitHub Release
- 当前 workflow 已固定安装 CUDA 11.8 版本栈（`requirements-dev-cu118.txt`，即 `torch==2.7.1+cu118` + `transformers==4.57.7`）

## 注意事项

- `torch + transformers` 打包后体积较大，建议使用目录式发布（已采用）。
- 建议 `device: auto`。若目标机无 NVIDIA CUDA 环境会自动回退到 CPU。
- Qwen3.5-35B-A3B 建议使用较新 `transformers`。若加载失败，先升级 `transformers` 后重试。
- 若你手动写了 `device: cuda` 或 `device: cuda:0`，但机器无 CUDA，程序现在会自动降级到 CPU 并输出告警。
