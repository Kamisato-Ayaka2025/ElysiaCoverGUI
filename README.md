# ElysiaCoverGUI

Windows 本地爱莉希雅翻唱工具。仅保留自训 **160000 步主模型**与已有 **350000 步浅扩散模型**的策略；扩散可在界面中开关。这里的 350000 对应 `model_350000.pt`，不是 35000。

Git 仓库包含 GUI 和推理编排源码；160000 主模型、350000 扩散模型及对应配置单独放在 [GitHub Release v1.0.0](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/tag/v1.0.0)。训练素材、歌曲、生成结果和第三方运行环境不随项目分发；只下载源码不能直接推理。

## 处理流程

1. Demucs v3 `UVR_Demucs_Model_1` 分离人声与伴奏：segment 10 秒、shifts 2、overlap 0.25。
2. `5_HP-Karaoke-UVR` 清理人声：window 320、aggression 25、batch 1，关闭 TTA、后处理和 high-end process。
3. So-VITS-SVC 4.1：G_160000、RMVPE、mean3、4 秒切片、noise 0.4、seed 1234。mean3 仅在连续有声区间对 F0 做三帧算术均值。
4. 可选浅扩散：350000 步、20 层、k_step 100、dpm-solver++、speedup 10、关闭 second encoding。
5. 按有效人声 RMS 匹配音量，再乘 1.180517552；统一峰值限制到 0.95。导出干声及带伴奏的 WAV 和 256 kbps MP3。

移调范围为 -12～+12 半音，伴奏同步移调。伴奏来自第一阶段的非人声分轨，第二阶段删掉的内容不会混回伴奏。合唱、弱人声仍可能被误删，输入歌曲需要试听确认。

## 环境与模型准备

需要 Windows、NVIDIA CUDA 显卡和含 `rubberband` 滤镜的 FFmpeg。项目使用两套独立 Python 环境，避免 SVC 与分离器依赖冲突。

已验证的现有环境组合：

| 用途 | Python | PyTorch | 其他依赖 |
| --- | --- | --- | --- |
| GUI / SVC | 3.8 | 1.13.1+cu117 | So-VITS-SVC 4.1 环境、Tkinter |
| UVR 分离 | 3.11 | 2.5.1+cu121 | audio-separator 0.47.1 |

先按 [So-VITS-SVC 项目](https://github.com/svc-develop-team/so-vits-svc) 与 [audio-separator 项目](https://github.com/nomadkaraoke/python-audio-separator) 的安装说明准备环境。上述版本是本机验证组合，并非全新机器的一键安装保证；不要在同一个环境里覆盖两套 PyTorch。

SVC 目录应包含其依赖和预训练组件：`pretrain/checkpoint_best_legacy_500.pt`、`pretrain/rmvpe.pt`、`pretrain/nsf_hifigan/model` 及对应配置。使用与本项目模型兼容的 SVC 4.1 代码。

UVR 模型目录需有 `ebf34a2db.th`、`5_HP-Karaoke-UVR.pth`，以及 `UVR_Demucs_Model_1.yaml`：

```yaml
models: ['ebf34a2db']
segment: 44
```

audio-separator 还需要对应模型元数据，首次加载可能联网获取。模型来源及许可请查阅 [UVR](https://github.com/Anjok07/ultimatevocalremovergui)。

当前策略要求 Elysia 单说话人（ID 0）、vec768l12、44.1 kHz。扩散来自已有模型，不是此次主模型重新训练出的配套扩散，效果需通过开关对照试听判断。

## 下载模型并放置

打开 [v1.0.0 的 Assets](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/tag/v1.0.0)，下载以下四个文件。不要只下载 GitHub 自动生成的 `Source code` 压缩包，它们不含权重。

| 文件 | 用途 |
| --- | --- |
| [G_160000.pth](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/download/v1.0.0/G_160000.pth) | 160000 步主模型，约 628 MB |
| [config.json](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/download/v1.0.0/config.json) | 主模型对应配置 |
| [model_350000.pt](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/download/v1.0.0/model_350000.pt) | 350000 步扩散模型，约 221 MB |
| [model_350000.yaml](https://github.com/Kamisato-Ayaka2025/ElysiaCoverGUI/releases/download/v1.0.0/model_350000.yaml) | 扩散模型对应配置 |

在源码根目录（`app.py` 旁）新建 `models` 文件夹，将四个文件直接放进去，不要再嵌套一层文件夹：

```text
ElysiaCoverGUI/
├── app.py
├── config.example.json
├── config.local.json
└── models/
    ├── G_160000.pth
    ├── config.json
    ├── model_350000.pt
    └── model_350000.yaml
```

`models/config.json` 是模型结构配置；根目录的 `config.local.json` 是本机路径配置，二者不能互相覆盖。`config.example.json` 已默认指向上述四个相对路径：新建本机配置后只需按实际情况修改运行环境路径。已有 `config.local.json` 的用户需将 `main_model`、`main_config`、`diffusion_model`、`diffusion_config` 更新为 `config.example.json` 中的对应值。

Release 同时提供 `SHA256SUMS.txt`。可用 `Get-FileHash models\G_160000.pth -Algorithm SHA256` 等命令逐项核对下载完整性。文件均为原始模型及对应配置；这个模型包不包含前述 UVR 分离模型、ContentVec、RMVPE、NSF-HiFiGAN 或 Python 环境，仍需完成环境准备。

## 配置与启动

```powershell
Copy-Item config.example.json config.local.json
```

编辑 `config.local.json`，填写运行环境、FFmpeg、UVR 和模型文件路径。相对路径以本项目目录为基准，也支持绝对路径；JSON 中 Windows 路径使用 `/` 或转义的 `\\`。本机配置已加入 `.gitignore`。

双击 `Start-ElysiaCover.bat`（需 Windows Python Launcher），或使用已配置的 SVC Python 执行 `app.py`。选择歌曲、移调和扩散开关后开始转换。

命令行示例（`python` 指 SVC 环境的解释器）：

```powershell
python worker.py --check
python worker.py --input "song.mp3" --pitch 0
python worker.py --input "song.mp3" --pitch 0 --no-diffusion
```

结果保存在 `outputs/<任务编号>/`。`job.json` 记录模型及参数，`pipeline.log` 记录各阶段日志，`separation.json` 记录分离权重及音频哈希。`--check` 检查主要文件存在性；GPU、依赖和模型兼容性仍需实际转换验证。输出磁盘至少预留 2 GiB，长歌曲需要更多空间。

## 开发与验证

`python -m unittest discover -s tests -v` 运行不需要 GPU 和权重的配置测试。真实验证建议用自己有权使用的短音频分别开启/关闭扩散，检查干声、伴奏版、移调及界面取消功能。固定曲目对照为《富士山下》（低音）与《TruE》（高音），歌曲不随仓库分发。

`app.py` 为 Tkinter 界面；`worker.py` 编排子进程、换声及混音；`blog_separator.py` 执行两阶段分离；`model_profiles.py` 固定唯一策略；`runtime_config.py` 解析路径。

模型、第三方软件和音频各自的授权独立于本仓库源码；使用和分享前请确认相应权限。不要提交歌曲、本机配置、日志或权重。
