# Beam Track Analyzer (Windows EXE)

基于本地 Hugging Face 视觉语言模型（首发支持 Qwen2.5-VL 系）分析多频带波束历程图，提取轨迹并筛选“仅在 2-3 个频带内出现”的目标轨迹。

## 功能

- 支持本地模型目录加载（`from_pretrained` + `safetensors`）。
- 支持一次输入多张图片，每张图片绑定一个 `band_id`。
- Prompt 与图片列表均由配置文件驱动，无需改代码。
- 程序内执行跨频带轨迹匹配与 2-3 频带筛选。
- 输出 JSON/CSV 结果与运行日志。

## 快速开始（源码方式）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m app.cli --config config/runtime.example.yaml --dry-run
```

正式运行：

```bash
python -m app.cli --config config/runtime.example.yaml
```

## 运行配置

复制并修改 `config/runtime.example.yaml`：

- `model.local_model_dir`：本地模型目录（含 safetensors / config / tokenizer 等）
- `input.images`：图片路径与 `band_id`
- `prompt.template_file` 与 `prompt.extra_instruction`
- `matching`：轨迹匹配阈值与频带数量筛选

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

## 注意事项

- `torch + transformers` 打包后体积较大，建议使用目录式发布（已采用）。
- 默认 `device: cuda`，若目标机无 NVIDIA CUDA 环境，请改为 `cpu` 或 `auto`。
