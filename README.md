# CondGen_DownStream_Unsupervised_Object_Discovery

本任务比较 **DINO、iBOT、MAE、CondGen 四个标准 ViT-B/16 encoder** 的无监督物体发现能力：冻结 encoder，以 TokenCut 为每张图片预测一个类别无关的边界框，按 CorLoc 评分，不训练下游网络。

**四个方法脚本均已就绪，使用本地 ViT-B/16 权重和最后一层 output 特征。不启用 `--no_hard`，采用 `max IoU >= 0.5`。VOC07、VOC12、COCO20K 的正式实验以及 CondGen tau=0.45/0.50 扩展实验均已完成。结果现在按 `outputs/<dataset>/<method>/tau_<value>/` 保存，队列元数据归档在 `outputs/_queues/`。**

## DINOv3-7B/16：27 项 tau 扫描（分片权重已验证，待启动）

新增 DINOv3 ViT-7B/16 评测入口，使用 `facebook/dinov3-vit7b16-pretrain-lvd1689m` 的本地 Hugging Face 分片权重，当前目录为 `/data1/pretrain_models`，记录 revision `b80367753773648a6793235ab9c65cdbb029506f`。现有 `transformers==4.56.1`、`safetensors` 和 `accelerate` 已满足要求，评测强制离线读取，不会联网下载替代权重；也支持合并后的单个 safetensors 文件。

计划评测 VOC07、VOC12、COCO20K × `tau={0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40}`，共 **27 项**。默认 GPU6/7 分别执行 14/13 项，可用 `GPU_IDS` 修改。**6/6 分片、完整预检以及 GPU6 上真实 7B 单图 TokenCut 已验证通过，不需要为运行而合并权重；27 项正式实验尚未启动。**

```bash
cd /data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
# 无副作用，查看27项及输出路径
bash scripts/queue_dinov3_7b_tau.sh --dry-run
# 检查源码、三套真实loader、6个权重分片及GPU；不加载7B模型
bash scripts/queue_dinov3_7b_tau.sh --preflight
# 完整预检通过后启动tmux队列；默认GPU_IDS=6,7
GPU_IDS=6,7 bash scripts/queue_dinov3_7b_tau.sh
tmux attach -t ruobei_uod_dinov3_7b
# 也可在前台运行
GPU_IDS=6,7 bash scripts/queue_dinov3_7b_tau.sh --foreground
```

[适配层](TokenCut/dinov3_adapter.py) 同时支持分片目录和单个 `.safetensors` 文件。两条路径使用同一个逐 tensor 加载器：先通过 Accelerate 在 meta 设备上初始化模型，再逐个读取权重到 CPU，最后移动到指定 GPU；加载前核对模型配置、tensor key/shape、FP32 dtype 和总参数量，加载后检查不存在未初始化参数。模型使用 SDPA，实际加载信息写入 `run_config.json`。模型原始输出为 `CLS + 4 register tokens + spatial patches`，适配层只移除 register tokens、保留 CLS，使 NCut 仍只删除一次 CLS 并只使用空间 patch。DINOv3 使用 eval 模式 RoPE，不启用训练时的随机 shift/jitter/rescale；其他图像归一化、原图分辨率、右侧/下侧 padding、二值图、`eps=1e-5`、`max IoU>=0.5` 协议与原四方法一致。

单项入口为 [eval_object_discovery_dinov3_7b.sh](scripts/eval_object_discovery_dinov3_7b.sh)，**默认使用物理 GPU6 且要求 CUDA**；队列默认使用 GPU6/7，每个进程独立加载一份模型。单项示例如下：

```bash
DATASET=VOC07 bash scripts/eval_object_discovery_dinov3_7b.sh 0.20
```

### tau=0.20/0.30 六项队列（已于 2026-11-11 18:01:30 启动）

专用入口为 [queue_dinov3_7b_tau020_030_gpu67.sh](scripts/queue_dinov3_7b_tau020_030_gpu67.sh)，配置见 [dinov3_7b_tau020_030_sweep.json](configs/dinov3_7b_tau020_030_sweep.json)。每项仍是独立进程、逐图 forward、逐图 NCut 和 CorLoc，**没有引入 batch 推理，也没有改变 FP32、原图分辨率或数据集划分**。

| GPU | tau | 串行顺序 | 每项图片数 |
| --- | --- | --- | --- |
| 6 | 0.20 | VOC07 → VOC12 → COCO20K | 5,011 / 11,540 / 19,817 |
| 7 | 0.30 | VOC07 → VOC12 → COCO20K | 5,011 / 11,540 / 19,817 |

运行记录：[20261111_180130_1060302](outputs/_queues/dinov3_7b_tau020_030_sweep/20261111_180130_1060302/)。tmux 会话为 `ruobei_uod_dinov3_7b_tau020_030`；启动后已确认两个 VOC07 进程均正常处理首批图片，双卡 `nvidia-smi` 显存均为 **26,564 MiB**，没有 OOM。六项结果尚未全部完成，不要重复启动同名队列。

```bash
bash scripts/queue_dinov3_7b_tau020_030_gpu67.sh --dry-run
bash scripts/queue_dinov3_7b_tau020_030_gpu67.sh --preflight
tmux attach -t ruobei_uod_dinov3_7b_tau020_030
```

显存检查使用每个数据集中 patch 数最多的样本：VOC 为 500×500、1024 patches，COCO 为 640×640、1600 patches。模型本身占用 **25.02 GiB**；最大 COCO 输入的 PyTorch 峰值预留为 **25.63 GiB**，`nvidia-smi` 为 **26.14 GiB / 47.38 GiB**，因此无需调整 batch size、降低输入分辨率或改变精度。完整数值见 [显存检查](outputs/_smoke/dinov3_7b_memory_check_20261111_175842.json)。

### 可选：将分片合并为单个 safetensors

直接读取分片目录就是正式支持的路径，无需转换。若后续需要单文件权重，可在全部分片下载完成后运行：

```bash
/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python \
    scripts/consolidate_dinov3_safetensors.py \
    /data1/pretrain_models \
    --output /data1/pretrain_models/dinov3_vit7b16.safetensors
```

[合并脚本](scripts/consolidate_dinov3_safetensors.py) 逐 tensor 流式复制，不一次性实例化 7B 模型或持有整份 state dict；它检查剩余磁盘空间，写入临时文件后逐 tensor 对比 SHA-256，全部一致才原子改名。源分片不会被修改或删除，默认拒绝覆盖已有合并文件。合并文件应与 `config.json` 放在同一目录。使用单文件启动预检和队列：

```bash
export DINOV3_WEIGHTS=/data1/pretrain_models/dinov3_vit7b16.safetensors
GPU_IDS=6,7 bash scripts/queue_dinov3_7b_tau.sh --preflight
GPU_IDS=6,7 bash scripts/queue_dinov3_7b_tau.sh
```

结果写入 `outputs/<dataset>/dinov3/tau_<value>/`，已有目录会拒绝覆盖。[队列脚本](scripts/queue_dinov3_7b_tau.sh) 单项失败后继续后续任务；[汇总脚本](scripts/summarize_dinov3_7b_queue.py) 只在 27 项全量结果、退出状态及模型/评测协议均通过时判定完成。完整参数记录见 [dinov3_7b_tau_sweep.json](configs/dinov3_7b_tau_sweep.json)。

准备阶段已通过三套真实数据 loader、GPU6/7 可见性、27 项 dry-run、worker 失败续跑和汇总模拟检查；[DINOv3 CPU 回归测试](tests/test_dinov3_adapter.py) 覆盖真实小模型的分片/单文件加载、流式合并逐值一致性、矩形前向、最终 LayerNorm、register token 移除和 NCut。**真实 7B 也已直接从 6 个分片完成加载和 GPU6 单图前向，687 个 tensor 全部通过检查，missing/unexpected/mismatched 均为空。** 证据见 [smoke 配置](outputs/_smoke/dinov3_7b_sharded_005625_tau0.20_20261111_175557/TokenCut-dinov3-vit_7b16_output_tau0.2/run_config.json) 和 [预测框](outputs/_smoke/dinov3_7b_sharded_005625_tau0.20_20261111_175557/TokenCut-dinov3-vit_7b16_output_tau0.2/005625_TokenCut_pred.jpg)。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    /data0/cuijiequan/env/miniconda3/envs/irepa/bin/python tests/test_dinov3_adapter.py
```

## 已有结果与耗时（2026-09-21 更新）

附录表现在同时包含 DINOv3 表中的 Agglomerative、Weakly-supervised 和 Self-supervised backbone 对照行；这些行的 ViT 列沿用原表结构，$\tau$ 记为 `--`。我们的四个 ViT-B/16 方法统一使用 B/16，DINO/iBOT/MAE 搜索七个 tau，CondGen 另增加 tau=0.45/0.50。

截至 2026-09-22，90/90 个正式实验均已完成：VOC07 30/30、VOC12 30/30、COCO20K 30/30；其中 DINO、iBOT、MAE 各有 tau=0.10--0.40 七项，CondGen 另有 tau=0.45/0.50 两项。所有结果均核验退出码、全量图数及评测协议一致，精确计数、来源和结果哈希见 [结果快照](reports/object_discovery_results_update_20260921.json)。

按每个方法、每个 tau、遍历一次完整数据集计一次实验：历史 wall 均值为 VOC07 **6.2 分钟**、VOC12 **18.7 分钟**、COCO20K **1.83 小时**。当前没有剩余评测任务；这些耗时可用于规划后续新增组合。详细计算见 [耗时报告](reports/object_discovery_runtime_update_20260921.md)。

## 定性对比样本

[object_discovery_compare](object_discovery_compare/) 保存三个数据集的 CondGen 最优样本和四方法对比样本。每个数据集有 `*_condgen_best` 与 `*_compare` 两组，各含 50 个按指标排序的样本；样本目录包含原图、已有评测框、重新核对过的 TokenCut 红色 mask、二值 patch mask、指标和源配置。`compare` 按 CondGen 相对 DINO、iBOT、MAE 的三项 IoU 差值之和排序。总体验证记录在 [validation.json](object_discovery_compare/validation.json)。

| 队列 / GPU | 状态 | 备注 |
|---|---|---|
| VOC07 / GPU6-7 | 已完成 | 20/20 |
| VOC12 / GPU6-7 | 已完成 | 20/20 |
| COCO20K / GPU6-7 | 已完成 | 20/20 |
| 高 tau / GPU6-7 | 已完成 | 24/24 |
| CondGen tau=0.45/0.50 / GPU4-5 | 已完成 | 6/6 |

## 定性可视化：DINOv3 风格红色 mask overlay

`TokenCut/visualizations.py` 原有 `visualize_predictions` 可画预测框，`main_tokencut.py` 现在还支持 `--visualize mask`：它把 `ncut()` 返回的 seed connected-component patch mask 最近邻展开到输入图像，并以 `0.3\,I+0.7\,[255,0,0]` 生成红色半透明 overlay。全量评测只保存框坐标，因此已有 `predictions.jsonl` 不能恢复 mask；可视化需要对选定图片重新推理。

例如用当前 CondGen 最佳阈值 `\tau=0.45`：

```bash
bash scripts/visualize_condgen_mask.sh \\
  /data0/datasets/dino_downstream/voc2007/VOCdevkit/VOC2007/JPEGImages/005625.jpg \\
  0.45
```

脚本使用 `condgen_vitb16.pth`、最后一层 `output` patch features、二值图和 `eps=10^{-5}`，不读取标注、不画 GT、不使用 bilateral solver 或其他后处理。`--resize 1024` 表示短边缩放到 1024 并保持纵横比，右侧和下侧只补齐到 16 的倍数；输出文件位于 `outputs/visualizations/.../*_TokenCut_mask_overlay.png`。这与 DINOv3 Figure 14 的“red overlay on input images at res. 1024”定性展示一致，但表格中的 CorLoc 仍使用全分辨率评测协议。

## CondGen tau=0.45/0.50 队列（6 项，GPU 4/5）

[queue_condgen_tau045_050_gpu45.sh](scripts/queue_condgen_tau045_050_gpu45.sh) 仅评测 CondGen ViT-B/16，共 VOC07、VOC12、COCO20K × tau={0.45,0.50} 六项。GPU4 执行 tau=0.45，GPU5 执行 tau=0.50；每卡三项，顺序为 VOC07 → VOC12 → COCO20K，当前评测进程退出后立即启动下一项。

**已于 2026-09-21 17:29（Asia/Shanghai）启动，现已 6/6 完成。** tmux 会话仍保留为 `ruobei_uod_condgen_tau045_050`；[队列归档目录](outputs/_queues/condgen_tau045_050_sweep/20260921_172952_2936493/) 保存日志和状态，六项实验结果分别位于 `outputs/<dataset>/condgen/tau_<value>/`。GPU4/5 分别完成 tau=0.45/0.50 在 VOC07、VOC12、COCO20K 上的全量评测；该队列在此前 84 项实验之外增加六项，累计计划 90 项。

沿用当前 TokenCut 协议：原图分辨率、最后一层 LayerNorm 后的 patch output、冻结 encoder、排除 CLS、二值图、eps=1e-5，保留 difficult/truncated 对象，max IoU>=0.5。COCO 使用固定 19,817 张 COCO20K，VOC12 为 11,540 张 trainval。

```bash
# 查看分配或检查三个数据集
bash scripts/queue_condgen_tau045_050_gpu45.sh --dry-run
bash scripts/queue_condgen_tau045_050_gpu45.sh --preflight
# 启动新队列（当前已运行，无需重复执行）
bash scripts/queue_condgen_tau045_050_gpu45.sh
tmux attach -t ruobei_uod_condgen_tau045_050
```

支持 `--foreground`。每项记录独立日志与预测，`manifest.tsv` 固定六项分配；`gpu4_status.tsv` / `gpu5_status.tsv` 记录退出码和起止时间。每卡自动更新汇总，全部结束后生成 `summary.csv/.md`。单项失败后继续剩余实验，整队列最终返回非零；完整评测要求全部六个组合均为全量结果且退出码为 0。启动时保存队列、CondGen 入口及汇总脚本副本供本次 worker 使用；队列独立锁防止重复启动。

配置：[condgen_tau045_050_sweep.json](configs/condgen_tau045_050_sweep.json)。验证：[六项调度检查](reports/condgen_tau045_050_queue_check.json)、[汇总完整性检查](reports/condgen_tau045_050_summary_check.json)、[真实启动核验](outputs/_queues/condgen_tau045_050_sweep/20260921_172952_2936493/launch_verification.json)。

## 新增 tau=0.35/0.40 队列（24 项）

[queue_object_discovery_high_tau_gpu67.sh](scripts/queue_object_discovery_high_tau_gpu67.sh) 覆盖 VOC07、VOC12、COCO20K × DINO、iBOT、MAE、CondGen × tau={0.35,0.40}。每个数据集 8 项，总计 24 项；GPU6 执行 tau=0.35 的 12 项，GPU7 执行 tau=0.40 的 12 项。每卡依次遍历 VOC07 → VOC12 → COCO20K，各卡内部串行运行。沿用四个方法入口、原图分辨率、output 特征、二值图、eps=1e-5、不启用 `--no_hard`、max IoU>=0.5。

**已于 2026-09-21 15:37 启动，与 COCO 基础队列并行共用 GPU6/7，现已完成 24/24。** [本次队列归档目录](outputs/_queues/high_tau_sweep/20260921_153706_2862331/) 保存 `queue.log` 和状态文件；24 项实验结果位于 `outputs/<dataset>/<method>/tau_<value>/`。此前 03:31 的等待任务在启动任何模型前已停止，[旧队列归档目录](outputs/_queues/high_tau_sweep/20260921_033125_2674782/) 保留，会话归档为 `ruobei_uod_high_tau_wait_033125`。

新队列使用 tmux 会话 `ruobei_uod_high_tau`；`manifest.tsv` 位于 `outputs/_queues/high_tau_sweep/<时间戳_PID>/`，包含 GPU、数据集、方法、tau 和 canonical 输出目录。逐项退出状态位于 `gpu6_status.tsv` / `gpu7_status.tsv`，汇总为 `summary.csv/.md`，保留全部组合。

每个高 tau worker 等待自己当前的评测进程退出，然后立即启动下一项；不检查 GPU 空闲状态，也不等待原队列结束。两个队列各自在每张卡上最多运行一个实验，因此并行期间每卡同时运行两个实验。高 tau 队列仅使用自己的 `high_tau_gpu67.lock` 防止重复启动，不获取基础队列的 `gpu6.lock` / `gpu7.lock`。已移除 `--after-sequence`；带有旧依赖标记的运行目录不能直接重启，需创建新运行目录。

```bash
# 只查看24项分配或检查三个数据集
bash scripts/queue_object_discovery_high_tau_gpu67.sh --dry-run
bash scripts/queue_object_discovery_high_tau_gpu67.sh --preflight
# 启动新队列（当前已启动，无需重复执行）
bash scripts/queue_object_discovery_high_tau_gpu67.sh
tmux attach -t ruobei_uod_high_tau
```

单项失败后继续其余项，整队列最终返回非零并保留失败记录。配置见 [high_tau_sweep.json](configs/high_tau_sweep.json)。[共享 GPU 调度检查](reports/high_tau_shared_gpu_check.json)验证了基础队列持锁时仍能启动、每卡 12 项按进程退出顺序执行，以及拒绝重复启动高 tau 队列。[切换记录](reports/high_tau_sharing_transition.json)记录旧等待任务与新运行的对应关系。[原计划及数据预检](reports/high_tau_queue_check.json)和[原模拟运行检查](reports/high_tau_worker_check.json)保留为历史记录。

## VOC12 与 COCO20K 数据及后续队列

数据准备脚本：[scripts/prepare_voc12_coco.py](scripts/prepare_voc12_coco.py)。本地压缩包保留在 `/data0/datasets/dino_object_discovery/`，解压到各自数据目录，项目通过软链接使用：

| 项目入口 | 实际数据位置 |
| --- | --- |
| `TokenCut/datasets/VOC2012/VOCdevkit` | `/data0/datasets/dino_object_discovery/voc2012/VOCdevkit` |
| `TokenCut/datasets/COCO/images` | `/data0/datasets/dino_object_discovery/coco2014/images` |
| `TokenCut/datasets/COCO/annotations` | `/data0/datasets/dino_object_discovery/coco2014/annotations` |
| `TokenCut/datasets/instances_train2014_sel20k.json` | `/data0/datasets/dino_object_discovery/coco2014/annotations/instances_train2014_sel20k.json` |

VOC12 使用检测 `trainval` 的 **11,540 张图片**；归档中的其他图片保留，评测只按该划分清单读取。COCO2014 `train2014.zip` 全量解压为 **82,783 张图片**，COCO 实验使用上游固定 **19,817 张 COCO20K**，不使用全部训练集，也不重新随机抽样。数据目录中的清单与 TokenCut 清单逐字节一致。

COCO 子集 JSON 在准备阶段统一生成，保留原标注与类别；后续队列直接读取，避免两张 GPU 同时构建缓存。COCO 使用 `--dataset=COCO20k --set=train`，读取软链接，不传 VOC 专用的 `--data_path`。维持上游忽略 `iscrowd` 的评分规则；固定子集每张图片都有非 crowd 目标。`datasets.py` 现在按 `image_id` 查文件名，消除子集 JSON 中图片顺序变化造成的命名错配，框转换与评分规则不变。

四个现有方法脚本新增 `DATASET=VOC12` / `DATASET=COCO20k` 选择，默认仍为 `VOC07`：

```bash
# 单个方法和tau示例
DATASET=VOC12 CUDA_VISIBLE_DEVICES=6 DEVICE=cuda \
    bash scripts/eval_object_discovery_dino_b16.sh 0.10
DATASET=COCO20k CUDA_VISIBLE_DEVICES=7 DEVICE=cuda \
    bash scripts/eval_object_discovery_condgen_b16.sh 0.30
```

每个新数据集队列均为 **DINO/iBOT/MAE/CondGen × tau={0.10,0.15,0.20,0.25,0.30}**，共 20 项，GPU6/7 各串行 10 项：

| 入口 | tmux 会话 | 结果目录 |
| --- | --- | --- |
| [queue_voc12_tau_gpu67.sh](scripts/queue_voc12_tau_gpu67.sh) | `ruobei_uod_voc12` | `outputs/_queues/voc12_tau_sweep/<时间戳_PID>/` |
| [queue_coco20k_tau_gpu67.sh](scripts/queue_coco20k_tau_gpu67.sh) | `ruobei_uod_coco20k` | `outputs/_queues/coco20k_tau_sweep/<时间戳_PID>/` |
| [queue_voc12_then_coco_tau_gpu67.sh](scripts/queue_voc12_then_coco_tau_gpu67.sh) | `ruobei_uod_voc12_then_coco` | 顺序日志在 `outputs/_queues/voc12_then_coco_tau_sweep/<时间戳_PID>/` |

**按 VOC12 → COCO 顺序启动整个 40 项任务，使用第三个入口**；它等待 VOC12 两卡全部结束且结果通过完整性检查，再启动 COCO。若 VOC12 任意实验失败，会先执行完 VOC12 队列剩余项，再停止后续 COCO 阶段。每项保留独立日志、退出码、预测、配置和结果，各数据集完成后生成 `summary.csv/.md`，每卡执行过程中生成 `summary_gpu6` / `summary_gpu7`。全部阈值分别保留，不自动选最优值。

```bash
cd /data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
# 查看40项顺序与分卡，不启动实验
bash scripts/queue_voc12_then_coco_tau_gpu67.sh --dry-run
# 只检查两套数据、依赖和loader，不启动实验
bash scripts/queue_voc12_then_coco_tau_gpu67.sh --preflight
# 准备完成后，执行这一条即可先VOC12、后COCO
bash scripts/queue_voc12_then_coco_tau_gpu67.sh
# 查看会话
 tmux attach -t ruobei_uod_voc12_then_coco
```

两个数据集的独立队列均支持 `--foreground`（阻塞直到完成）和 `--preflight`。共用实现为 [queue_object_discovery_tau_gpu67.sh](scripts/queue_object_discovery_tau_gpu67.sh)；它用项目内 GPU 锁防止这两个新队列同时占用 6/7 卡，锁不代表其他项目的 GPU 使用情况。普通启动前应确保这两张卡可用。协议记录见 [voc12_tau_sweep.json](configs/voc12_tau_sweep.json) 和 [coco20k_tau_sweep.json](configs/coco20k_tau_sweep.json)。准备检查通过后，已于 2026-09-21 01:21（Asia/Shanghai）启动这两个数据集的顺序队列，会话为 `ruobei_uod_voc12_then_coco`。启动时已确认 GPU 6/7 分别执行 VOC12 DINO tau=0.10/0.15，均使用全部 11,540 张图片并正常推进；VOC12 全部成功后自动进入 COCO。[顺序队列归档](outputs/_queues/voc12_then_coco_tau_sweep/20260921_012108_2663376/)和[独立队列归档](outputs/_queues/voc12_tau_sweep/20260921_012115_2663798/)保存日志，实验结果位于 `outputs/VOC12/<method>/tau_<value>/` 和 `outputs/COCO20k/<method>/tau_<value>/`。

当前环境已补齐 `pycocotools==2.0.11`（`pip --no-deps`，没有变更其他包）。可重复检查数据准备和真实 loader：

```bash
EVAL_PYTHON=/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python
"$EVAL_PYTHON" scripts/prepare_voc12_coco.py --verify-only \
    --report reports/voc12_coco_recheck.json
"$EVAL_PYTHON" scripts/check_voc12_coco.py
```

2026-09-21 准备验收已通过：

- [数据报告](reports/voc12_coco_preparation.json)：完整解压后再次逐字节比对压缩包；全部 17,125 张 VOC 图片/XML 和选中的 19,817 张 COCO 图片通过解码、尺寸与标注检查。COCO 子集含 145,654 个标注，其中 143,951 个非 crowd；没有缺失图片或无有效非 crowd 目标的图片。VOC 原始 XML 中两处小数框不属于检测 trainval，未改动标注。
- [真实 loader 检查](reports/voc12_coco_loader_check.json)和[队列预检](reports/voc12_coco_queue_preflight.log)：两套实际加载数量、COCO ID 集合和软链接可用；首、中、末样本的图片与标注对应正确。
- [CPU 全流程检查](reports/voc12_coco_smoke_check.json)：DINO 在每个数据集各评测前 2 张原图，均计分 2 张、跳过 0 张，完成权重加载、特征提取、NCut、IoU 评分与结果保存。仅用于验证运行链路，不能作为全量 CorLoc 结果。
- [队列检查](reports/voc12_coco_queue_check.json)：确认每个数据集 20 个唯一组合、每卡 10 项；模拟验证单项失败后继续、VOC12 完成后才启动 COCO，以及拒绝将部分数据结果当作完整评测。另有 [12 项 CPU 回归测试](reports/tests_voc12_coco.log)全部通过。

本轮脚本、配置和代码哈希见 [voc12_coco_changes_manifest.json](reports/voc12_coco_changes_manifest.json)。

## GPU 6/7 排队实验（2026-09-21）

启动脚本：[scripts/queue_voc07_tau_gpu67.sh](scripts/queue_voc07_tau_gpu67.sh)。任务为 DINO、iBOT、MAE、CondGen × `tau={0.10,0.15,0.20,0.25,0.30}`，共 20 项，全部使用 VOC07 trainval 5,011 张、原图分辨率、output 特征、二值图、`eps=1e-5`。这是显式阈值扫描；单独运行方法脚本时仍默认 `tau=0.2`。

| 方法脚本 | checkpoint |
| --- | --- |
| [eval_object_discovery_dino_b16.sh](scripts/eval_object_discovery_dino_b16.sh) | `dino_vitb16.pth` |
| [eval_object_discovery_ibot_b16.sh](scripts/eval_object_discovery_ibot_b16.sh) | `ibot_vitb16.pth` |
| [eval_object_discovery_mae_b16.sh](scripts/eval_object_discovery_mae_b16.sh) | `mae_vitb16.pth` |
| [eval_object_discovery_condgen_b16.sh](scripts/eval_object_discovery_condgen_b16.sh) | `condgen_vitb16.pth` |

四脚本继承 `CUDA_VISIBLE_DEVICES`，接收位置参数 tau 或 `TAU` 环境变量，并支持 `OUTPUT_DIR`、`DEVICE`。例如：

```bash
CUDA_VISIBLE_DEVICES=6 DEVICE=cuda bash scripts/eval_object_discovery_dino_b16.sh 0.10
# 仅查看20项分配，不启动
bash scripts/queue_voc07_tau_gpu67.sh --dry-run
# 开始新的20项实验；已有同名tmux会话时拒绝重复启动
bash scripts/queue_voc07_tau_gpu67.sh
# 查看当前会话
 tmux attach -t ruobei_unsupervised_object_discovery
```

**本次 20 项已全部成功完成，无需重复启动。** [队列归档目录](outputs/_queues/voc07_tau_sweep/20260921_000514_2652261) 的 `manifest.tsv` 固定记录 20 项分配，`gpu6.log/gpu7.log` 为两个串行队列日志，每卡退出状态记录在 `gpu6_status.tsv/gpu7_status.tsv`。每项实验的 `discovery.log` 和结构化结果位于 `outputs/VOC07/<method>/tau_<tau>/`。

每个实验完成后更新本卡的 `summary_gpu6.csv/.md` 或 `summary_gpu7.csv/.md`；两卡全部结束后自动写入 `summary.csv` 和 `summary.md`，保留全部 20 个结果，不自动选择最佳 tau。也可随时手动汇总：

```bash
/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python scripts/summarize_voc07_queue.py \
    outputs/_queues/voc07_tau_sweep/20260921_000514_2652261
```

两个队列分别绑定 GPU 6/7，同一卡不并发；设置 OMP/OpenBLAS/MKL 各 1 个 CPU 线程，避免多个 NCut 特征分解过量创建线程。每个进程显式 `--device=cuda`。此前论文表格的固定 `tau=0.2` caption 对应单阈值结果；本次扫描保留全部阈值，若之后报告最优值，需要同时更新 caption 并注明是在评测集上选阈值。

## 单独运行方法脚本

```bash
bash /data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery/scripts/eval_object_discovery_dino_b16.sh
```

脚本风格沿用 [eval_video_seg_condgen_s.sh](../dino/scripts/eval_video_seg_condgen_s.sh)：`set -euo pipefail`、继承调用方 GPU 设置、`OMP_NUM_THREADS=1`、绝对工作目录、显式 Python/数据路径、按数据集/方法/tau 的固定结果目录，以及 `python -u ... 2>&1 | tee ...` 日志。

默认使用 `/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python`，读取 `/data0/datasets/dino_downstream/voc2007` 下 VOC07 **trainval 的全部 5,011 张图片**，加载 `../encoder/dino_vitb16.pth` 的 `model` 权重。主程序默认 `--device=auto`，有可用 CUDA 时使用 GPU，否则使用 CPU；可手动加 `--device=cuda` 要求 CUDA。

输出示例：

```text
outputs/VOC07/dino/tau_0.20/
├── discovery.log
└── VOC07_trainval/
    └── TokenCut-dino-vit_base16_output_tau0.2/
        ├── run_config.json       # 参数、checkpoint 信息和实际协议
        ├── predictions.jsonl    # 逐图预测框、max IoU 和成功标记
        ├── preds.pkl            # 上游格式的预测框
        ├── results.json         # 精确 CorLoc、成功数和实际分母等
        └── results.txt          # 兼容原入口的文本结果
```

`results.json` 的 `corloc_percent` 是百分数；同时记录 `successful_images`、`evaluated_images`、`skipped_images`、`dataset_images`、`selected_images` 和 `subset_evaluation`。同一实验目录已有 `run_config.json` 时会拒绝覆盖，重复运行应使用新的 tau 或明确创建新的实验目录。进程失败会由 `pipefail` 返回非零状态，日志保留在对应的 `outputs/<dataset>/<method>/tau_<value>/` 目录。

如需先验证少量真实图片，可直接调用同一入口：

```bash
cd /data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery/TokenCut
OBJECT_PYTHON=/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python
OMP_NUM_THREADS=1 "$OBJECT_PYTHON" -u main_tokencut.py \
    --encoder=dino --arch=vit_base --patch_size=16 \
    --checkpoint_key=model \
    --pretrained_weights=/data0/cuijiequan/zhangruobei/CondGen_DownStream/encoder/dino_vitb16.pth \
    --data_path=/data0/datasets/dino_downstream/voc2007 \
    --dataset=VOC07 --set=trainval --which_features=output \
    --tau=0.2 --eps=1e-5 --device=cpu --max_images=2 \
    --output_dir="../outputs/_smoke/smoke_dino/$(date +%Y%m%d_%H%M%S)"
```

`--max_images` 只用于调试，按数据集顺序选取前 N 张，并在结果中标记为子集评测；正式脚本不设置该参数。少量图片的 CorLoc 不能当作全量性能。

## 四个 encoder 与已实现的代码改动

四份标准 ViT-B/16 权重都放在任务目录上一级的 `encoder/` 中，均为已有文件的软链接，实际容器均为 `checkpoint["model"]`。

| 方法 / `--encoder` | `--pretrained_weights` 文件 | 原始 tensor 数 | 排除的 head tensor 数 | backbone tensor 数 |
| --- | --- | ---: | ---: | ---: |
| `dino` | `../encoder/dino_vitb16.pth` | 152 | 2 | 150 |
| `ibot` | `../encoder/ibot_vitb16.pth` | 162 | 12 | 150 |
| `mae` | `../encoder/mae_vitb16.pth` | 152 | 2 | 150 |
| `condgen` | `../encoder/condgen_vitb16.pth` | 152 | 2 | 150 |

表中路径相对于任务根目录；模板使用绝对路径，避免从 `TokenCut/` 运行时的相对路径歧义。评测其他方法时直接运行对应的四个方法脚本；每个脚本已经绑定对应的 `--encoder` 和 checkpoint。四个方法都保持 `--arch=vit_base --patch_size=16 --checkpoint_key=model --which_features=output`，不要把 MAE 方法名当作另一个网络结构。

| 文件 | 已实现的行为 |
| --- | --- |
| [TokenCut/networks.py](TokenCut/networks.py) | 支持四方法、本地 checkpoint 路径和 `checkpoint_key`；过滤 `head.*` 后严格加载 backbone，不为本地权重失败自动下载替代权重 |
| [TokenCut/main_tokencut.py](TokenCut/main_tokencut.py) | 增加 `output` 特征、本地权重/数据参数、设备选择、小批验证、分模型/阈值输出和结构化结果记录 |
| [TokenCut/datasets.py](TokenCut/datasets.py) | 支持 `--data_path` 指定 VOC 数据根目录；COCO20K 校验固定子集 ID、按 ID 查文件名；数据读取和可视化共用同一路径 |
| [TokenCut/object_discovery.py](TokenCut/object_discovery.py) | 保留原 NCut 算法及带 CLS 的输入约定，增加 token 数与网格检查，防止重复删除 CLS |
| [scripts/eval_object_discovery_dino_b16.sh](scripts/eval_object_discovery_dino_b16.sh) | DINO ViT-B/16、VOC07 trainval、固定默认阈值的正式运行模板 |

四模型统一复用 `TokenCut/dino/vision_transformer.py` 的 `vit_base(patch_size=16, num_classes=0)`：12 blocks、12 attention heads、768 维、MLP ratio 4、qkv bias、LayerNorm eps `1e-6`，骨干参数量均为 **85,798,656**。无需改变网络结构。

`output` 路径调用 `model.get_intermediate_layers(image, n=1)[0]`，获取最后一个完整 block 经最终 LayerNorm 的 `[CLS + patches]` 输出；原版 `ncut()` 内部只删除一次 CLS，图中只使用 patch。最后层 output 不是 attention 的 key，也不是 `model(image)` 返回的 CLS。原入口仍保留 `--which_features=k` 默认值；**本任务模板显式传入 `--which_features=output`**，其余 TokenCut 主要算法参数沿用默认。

MAE 同样对完整 patch 编码，无随机 masking、不运行 decoder。四模型使用同一份 DINO ViT 矩形位置编码插值（bicubic、`+0.1` scale offset）。最终 LayerNorm 后的 output 是本项目统一选择，DINOv3 D.4 没有进一步公布所有实现细节。

## 固定评测协议

[configs/voc07_protocol.json](configs/voc07_protocol.json) 是协议记录，**主程序不会读取这个 JSON**；实际参数由命令行和脚本决定。

- VOC07 `trainval` 全部 5,011 张，RGB、ImageNet mean/std，保持原图分辨率，不做固定尺寸缩放或中心裁剪。
- 在归一化后图像的右侧和底部补 0 到 16 的倍数，预测框裁剪回原图边界。
- TokenCut 对 patch 做 L2 归一化，二值图强边条件为 `cosine > tau`，**固定 `tau=0.2`、弱边 `eps=1e-5`**；保留原 NCut 求解、均值二分、seed 连通区域和 4 连通规则。
- **默认不传 `--no_hard`**，保留 difficult/truncated 目标。若启用该选项，会删除这些框并跳过仅含此类目标的 1,461 张图片，使分母从 5,011 变为 3,550，属于另一个协议。
- 沿用上游 VOC 坐标转换：只对 xmin/ymin 减 1，xmax/ymax 保持，IoU 使用半开坐标，不额外加 1。
- **成功条件为与任意真值框的 `max IoU >= 0.5`**，CorLoc = 成功图数 / 实际评测图数 × 100%。DINOv3 D.4 原文字为严格 `> 0.5`；这里按要求沿用 TokenCut 的 `>=`。

**单方法脚本默认 `tau=0.2`；本次队列按上述五个 tau 显式扫描，不自动选取最优阈值。** 扫描结果与固定 `tau=0.2` 的结果应明确区分。在 trainval 上取最大值需明确标为“本评测集上的最佳 tau”，不能称作独立调参后的 test 性能。

本任务的 ViT-B/16、trainval 和 output 是显式比较设置；TokenCut 上游入口原默认是 ViT-S/16、train 和 key，因此本模板不是逐项复刻上游默认命令。DINOv3 Figure 14 的 DINO-B/16 VOC07 **60.1**、VOC12 **64.4**、COCO20K **50.5** 只作为对应论文协议参照；上游 README 的 key 特征成绩也不应直接当作本项目 output 协议的预期值。

## VOC2007 与准备检查

数据已解压在原始路径，保留归档、JPEG 与 XML 检测标注，无需转成分类目录：

```text
/data0/datasets/dino_downstream/voc2007/
├── archives/
│   ├── VOCtrainval_06-Nov-2007.tar
│   └── VOCtest_06-Nov-2007.tar
└── VOCdevkit/VOC2007/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/Main/{train,val,trainval,test}.txt
```

`--data_path` 指向包含 `VOCdevkit/` 的 `/data0/datasets/dino_downstream/voc2007`。原先的 `TokenCut/datasets/VOC2007/VOCdevkit` 软链接也保留，未传 `--data_path` 时仍可使用上游默认路径。

| 划分 | 图片数 | 用途 |
| --- | ---: | --- |
| train | 2,501 | 已校验 |
| val | 2,510 | 已校验 |
| **trainval** | **5,011** | **正式评测划分** |
| test | 4,952 | 已解压备用 |
| 全部唯一图片 | 9,963 | 对应 9,963 个 XML |

准备阶段已逐张解码全部 JPEG、解析全部 XML，核对图像宽高、框边界及划分关系。trainval 有 15,662 个目标框，test 有 14,976 个；归档 SHA-256 和检查见 [voc2007_preparation.json](reports/voc2007_preparation.json)。

四份真实权重已严格加载，并通过 CPU FP32 的 224×224、224×320、320×224 前向检查，去掉 CLS 后 shape 分别为 `[1,196,768]`、`[1,280,768]`、`[1,280,768]`，输出均为有限值，详见 [encoder_check.json](reports/encoder_check.json)。这些准备检查不是正式 CorLoc benchmark。

本轮修改后，又通过真实 `main_tokencut.py` 入口在 CPU 上分别评测四模型的 VOC07 trainval 前两张原图，完成本地权重加载 → output 特征 → NCut → 框与 IoU → 结果文件的全流程检查。四次运行均从 `/tmp` 启动并使用显式数据路径，均计分 2 张、跳过 0 张，配置确认 `tau=0.2`、`no_hard=False`、`IoU >= 0.5`，逐图框位于原图边界内。记录见 [adaptation_smoke_check.json](reports/adaptation_smoke_check.json)。环境与数据入口的 6 项检查也全部通过：[setup_check_adapted.json](reports/setup_check_adapted.json)。未启动全量 GPU 评测。

[tests/test_tokencut_adaptation.py](tests/test_tokencut_adaptation.py) 的 8 项 CPU 回归测试全部通过，覆盖完整最后层与 norm、q/k/v 上游一致性、hook 清理、矩形 NCut 与原版逐值一致、CLS/网格检查、严格权重加载以及本地路径失效时禁止下载回退。

```bash
cd /data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
EVAL_PYTHON=/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python

# 已解压数据的全量核查
"$EVAL_PYTHON" scripts/prepare_voc2007.py --verify-only \
    --report reports/voc2007_recheck.json

# 四份真实权重的严格加载与 CPU 前向检查
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" scripts/check_encoders.py

# 环境、VOC trainval 入口和 NCut 检查
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" scripts/check_setup.py

# 适配代码的 CPU 回归测试
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" tests/test_tokencut_adaptation.py
```

## 环境与来源

现有环境为 Python 3.10.21、PyTorch 2.7.0+cu126、torchvision 0.22.0+cu126、SciPy 1.15.3、NumPy 2.2.6、Pillow 11.3.0、scikit-image 0.25.2、transformers 4.56.1，并已安装 safetensors 与 accelerate。准备阶段环境报告见 [setup_check.json](reports/setup_check.json)。VOC 主路径无需安装上游历史 `requirements.txt` 的全部依赖；COCO 所需 `pycocotools==2.0.11` 已补齐，见 [coco_environment.json](reports/coco_environment.json)。

- 任务说明：[Unsupervised Object Discovery.md](../readme/Unsupervised%20Object%20Discovery.md)。
- DINOv3 论文：`/data0/cuijiequan/zhangruobei/related paper/2508.DINOv3.pdf`，§6.1.4、Figure 14、Appendix D.4；[D.4 文本摘录](reports/dinov3_D4_excerpt.txt)。
- 上游：[YangtaoWANG95/TokenCut](https://github.com/YangtaoWANG95/TokenCut)，基础快照 `fed52cd5b60891baefd8ec7110dafa73be816ee1`（2023-03-30）。完整仓库和 [LICENSE](TokenCut/LICENSE) 保留在 [TokenCut/](TokenCut/)。
- [source_manifest.json](reports/source_manifest.json) 记录初次克隆时的上游源码与 SHA-256；上表文件现已作本地适配，因此不能将初次清单视作修改后的文件校验值。
- 修改后的文件哈希见 [local_changes_manifest.json](reports/local_changes_manifest.json)，与固定上游快照的精确差异见 [tokencut_local_changes.patch](reports/tokencut_local_changes.patch)。每次正式运行的 `run_config.json` 也记录当时的源码和 checkpoint 哈希。

VOC12 与 COCO20K 的数据、软链接和队列现已接入，详见本文对应部分。
