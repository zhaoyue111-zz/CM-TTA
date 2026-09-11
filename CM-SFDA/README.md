# VoxTell P0 SFDA（方案 A：只训练 soft prompt）

该目录是 CM-TTA 的 VoxTell SFDA 版本。启动时 Qwen 文本编码器只把
`--prompt` 编码成一个初始向量；随后 Qwen、图像编码器、`project_text_embed`、
Transformer Decoder、`project_to_decoder_channels` 和 mask decoder 全部冻结，
唯一进入优化器的是从初始向量复制出的 `soft_prompt_embedding`。教师端只保存
这个向量的 `teacher_soft_prompt` EMA，不创建第二个 VoxTell 网络。

适配时使用 P0 `train_cases` 图像，不读取训练标签；teacher soft prompt 产生置信度
pseudo-label，weak/strong 与多视图强度增强用于一致性和视图筛选。测试时才读取
P0 `test_cases` 标签。

数据目录必须包含：

```text
data_dir/
├─ images/P0/<case_name>.nii.gz
├─ labels/P0/<case_name>.nii.gz
└─ worst_zeroshot_split_p0/worst_zeroshot_split.json
```

`worst_zeroshot_split.json` 必须包含 `train_cases` 和 `test_cases`。训练阶段只读取 `train_cases` 对应的图像，不读取标签；测试阶段读取 `test_cases` 对应的图像和标签。无论命令行如何设置，代码都只使用 P0 和上述固定 split 文件。

运行示例：

```powershell
python run_sfda_voxtell.py `
  --data_dir D:\path\to\data `
  --voxtell_root D:\path\to\VoxTell_from_disk `
  --model_dir D:\path\to\VoxTell_from_disk\model `
  --prompt prostate `
  --epochs 5
```

默认路径也可通过 `VOXTELL_ROOT`、`VOXTELL_MODEL_DIR` 环境变量覆盖；命令行参数优先，
路径不存在时程序会给出明确错误。建议调试时加
`--record_soft_prompt_grad_norm`，输出每次更新的 soft prompt 梯度范数。
CUDA AMP 的初始梯度 scale 默认为较稳妥的 1024，可通过 `--amp_init_scale` 调整；
偶发溢出会跳过该次参数及 teacher EMA 更新、自动降低 scale，并在日志中计入
`skipped`，连续 8 批溢出才会作为持续数值异常终止。

伪标签质量支持 `--quality_mode cac|purity|completeness|tse`。默认仍为 `cac`，完全
保留原有 CAC 消融；使用 TSE 的示例为：

```powershell
python run_sfda_voxtell.py `
  --data_dir D:\path\to\data `
  --voxtell_root D:\path\to\VoxTell_from_disk `
  --model_dir D:\path\to\VoxTell_from_disk\model `
  --prompt prostate `
  --quality_mode tse `
  --quality_config configs\tse.json `
  --w_quality 0
```

当前 TSE/purity/completeness 模式强制 `--w_quality 0`，质量指标只用于视图评价；
不会把 TSE 加入训练损失。CAC 模式继续使用 `--w_cac`（`--w_contrast` 是兼容别名）。
视图排序仍沿用原流程的 quality + entropy rank fusion，并列质量使用平均排名。

SAAF 视图选择使用冻结的初始 Qwen "liver" embedding 和 VoxTell 第一层
Transformer cross-attention（逐 head/token、log-attention median/MAD evidence），不使用
prototype/memory。命令为：

```bash
python run_sfda_voxtell.py --data_dir /path/to/data \
  --voxtell_root /path/to/VoxTell --model_dir /path/to/model \
  --prompt prostate --quality_metric saaf --quality_mode cac --w_quality 0
```

若某病例所有候选视图的 attention 或 SAAF 输入无效，该病例本次更新会跳过并记录原因；
无效视图不会静默回退到 CAC。SAAF 诊断写入 `saaf_diagnostics.json/csv`。

也可以直接使用仓库内的入口脚本（参数依次为 `DATA_DIR VOXTELL_ROOT MODEL_DIR`
以及可选的 prompt 和输出目录）：

```bash
bash train_cac.sh /path/to/data /path/to/VoxTell /path/to/model prostate results_/cac
bash train_tse.sh /path/to/data /path/to/VoxTell /path/to/model prostate results_/tse
# Explicit historical CAC-loss baseline (w_cac=1), if needed:
bash train_cac_baseline.sh /path/to/data /path/to/VoxTell /path/to/model prostate results_/cac_baseline
```

`train_cac.sh` is the fair view-selection-only CAC configuration and explicitly
passes `--w_cac 0`; `train_tse.sh` likewise passes `--w_quality 0`. Both launchers
resolve `run_sfda_voxtell.py` and `configs/tse.json` relative to their own directory,
so they are safe to call from any working directory. The legacy `train.sh` is
disabled to prevent accidentally running an implicit CAC baseline; choose one of
the two explicit launchers above (an original `w_cac=1` run must be named and
invoked separately as a baseline).

TSE 启动适配前，用冻结的初始 VoxTell、固定的单 prompt 和全部无标签 train cases
做一次 prototype 预扫描。每个病例先按现有 nnUNet 预处理得到完整非零区域，再右侧
padding 到 patch 网格，以确定性的无重叠 patch 覆盖完整病例；padding 区域由 valid
mask 排除。复用 `project_bottleneck_embed` 输出
`(B,H,W,D,C)`；高前景概率、高文本相似度且跨强度视图稳定的 voxel 聚合到前景
prototype，低概率、低相似度且稳定的 voxel 聚合到背景 prototype。病例级 seed
sum/count 会被保留，评价某病例时只聚合其他病例（严格 leave-one-case-out）。每个
病例和数据集的 seed 数/占比、有效病例数及相似度直方图/矩/分位数写入
`prototype_diagnostics.json`；全局无有效前景或背景 seed，或没有任何有效 LOO 原型时
会直接报错。
当前增强只有强度变换，空间逆变换为恒等；以后加入翻转/仿射时必须先逆变换再聚合。
阈值、特征 hook、seed 视图数、温度和 epsilon 全部位于 `configs/tse.json`。

原型和 evidence map 默认停止梯度；SAAF 仅用于无梯度视图选择（固定 `w_quality=0`）。
空 seed、空预测、空 evidence 均回退为有限的 0 分。prototype memory 随 checkpoint
保存和加载，并在已初始化的 DDP 进程组中按病例同步。v1 保留 prototype 轴但只允许
一个前景和一个背景 prototype，便于后续扩展多 prototype。
多视图排序使用无梯度前向，排序后只对选中的视图重新前向并保留反向图；默认配置
9 个候选视图只为最终选中的 1 个视图保存 3D 网络激活，以控制训练显存。

默认每 5 个 epoch（以及最后一个 epoch）在完整 test split 上评估，可用
`--eval_interval` 修改间隔，例如 `--epochs 20 --eval_interval 5` 会在第
5、10、15、20 轮验证。每次验证都会原子更新输出目录下的
`epoch_metrics.json`。文件中的 `epochs` 按轮保存 `cases` 和 `average`：每个病例
及病例宏平均均包含 Dice、IoU、Recall、Precision。其中 IoU 只计算目标前景，
平均 IoU 是各病例前景 IoU 的宏平均；Recall/Precision 也以目标前景为正类。最后一轮另外保存 `results.json`、
NPY/NIfTI 预测以及 ASSD/HD95；前几轮不保存 3D 预测，以控制磁盘占用。

输出还包括 `last.pt`；每轮测试前会先更新它，checkpoint 使用
`voxtell-sfda-prompt-tse-v4`，并继续接受旧 v1/v2/v3 checkpoint。
若训练已完成而评估中断，可用 `--eval_only --checkpoint <last.pt>` 只重新评估，
不会继续更新 soft prompt。

轻量检查（不加载真实 VoxTell 权重）：

```powershell
D:\anaconda\python.exe -m unittest discover -s CM-SFDA\tests -p "test_*.py" -v
```

带 GT 的质量审计是独立脚本。train split 不参与 SAAF（仅 legacy TSE 模式构建 prototype）；
test/validation 的 GT 只在完整体积 sliding-window 推理、evidence 融合和所有质量选择结束后
用于离线指标：

```powershell
python evaluate_quality_metrics.py `
  --data_dir D:\path\to\data `
  --voxtell_root D:\path\to\VoxTell_from_disk `
  --model_dir D:\path\to\VoxTell_from_disk\model `
  --prompt prostate `
  --quality_metric saaf --quality_mode cac `
  --w_quality 0
```

输出 `quality_audit.json`，包含 confidence、entropy、consistency、CAC、purity、coverage、SAAF
（并保留 completeness/TSE 兼容别名）与逐视图真实 Dice 的每病例 Spearman 宏平均及全局补充相关性，
并对每个滑窗 patch 计算质量指标与候选 view GT Dice 的 Spearman；同时输出 CAC/SAAF 单独选择、
CAC+entropy/SAAF+entropy 选择（以及 purity/coverage 单独选择）的完整体积 Dice、
patch-level selected Dice、patch-best Dice 和非负 patch-oracle gap。固定 view 的最佳完整体积
Dice 单独记为 `best_fixed_view_dice`，不参与 patch gap。汇总包含 mean selected Dice、
mean patch-oracle gap、best-fixed-view mean Dice、有效病例数和有效 patch 数。evidence 原图视图还输出空间 Dice、AUROC、AUPRC、均值、
标准差和分位数；空 GT/单类 GT 作为无效病例单独记录。默认保存前 5 个病例的 image、
GT、预测和 evidence 轴向切片可视化。

审计推理直接复用 VoxTell predictor 的 `pad_nd_image`、滑窗 slicer、Gaussian
importance map 和 logits 融合顺序：每个候选增强视图先在 patch 上得到 logits，完整体积
先按 Gaussian 权重融合 logits，最后一次 sigmoid。evidence 与 text similarity 使用完全
相同的空间权重。视图选择也按每个滑窗 patch 独立进行，再融合被选中的 logits；候选视图
采用与训练相同的随机 scale、offset 和 Gaussian noise，并由固定 seed 重现。

注意：当前 `CM-SFDA` VoxTell 分支没有原始 2D CM-TTA 中的 short prompt memory，
也没有 LSPM、DSPU 实现；因此本次没有伪造这些组件。已有 teacher prompt EMA、
伪标签生成、分割损失和推理流程均未改变。
