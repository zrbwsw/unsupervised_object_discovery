# Object discovery comparison

三个数据集各有 condgen_best、compare 两组，每组 50 张，共 300 个样本目录。
固定 tau：DINO=0.30、iBOT=0.15、MAE=0.35、CondGen=0.45。

condgen_best 按 CondGen 的 max IoU 降序；compare 按三项 (CondGen IoU − 基线 IoU) 之和降序。
这是使用标注指标挑选的定性示例；IoU 是框指标，不是 mask 指标，整图框也可能排在前列。
详见各目录 README.md、index.csv、all_scores.csv。原图、红框、红色 mask 及原始数值材料均保存在样本目录中。

另外，[VOC07_condgen_best_1024](VOC07_condgen_best_1024/) 保存了用户指定的 24 张 CondGen 定性样本。
这些样本用 `tau=0.45`、ViT-B/16、patch size 16 在短边 1024 的输入上重新推理，分别提供原始
nearest mask overlay 和仅用于显示的 bilinear overlay；该目录的框图用于后续与其他方法的定性对比。

完整 50 张 1024 定性图位于 [VOC07_condgen_best_1024_all50](VOC07_condgen_best_1024_all50/)，
附录拼图使用其中的 `mask_overlay_bilinear_alpha0.45.png`，输出为
`condgen_overlay_bilinear_alpha0.45_concat.pdf`。拼图脚本支持 `--ranks` 指定排名、
`--label-template` 控制标签文字，以及 `--font-size-pt`、`--dpi`、`--row-height-in`、
页面尺寸和边距参数。
mask 使用与全量评测相同的原分辨率推理，重算框必须匹配既有框，未额外缩放到 1024 或进行分割后处理。

## 查看

打开各目录 contact_sheet.jpg 快速浏览；进入样本目录查看 overview.jpg。
compare 样本的四方法子目录各含独立预测框、红色 mask、二值 mask、patch mask、eigenvector、源评测配置与重推理记录。
validation.json 记录导出完成状态和核验结果。selection.json 记录筛选，render_manifest.json 记录需要重新推理的去重任务。

## 复现

使用项目 scripts/prepare_object_discovery_compare.py --root NEW_DIRECTORY 生成索引、原图和预测框（拒绝覆盖已有目录）。
然后对四个 method 各运行 scripts/render_object_discovery_compare.py --manifest NEW_DIRECTORY/render_manifest.json --method METHOD --device cuda。
最后运行 scripts/finalize_object_discovery_compare.py --root NEW_DIRECTORY 生成拼图并验证。使用现有 irepa Python 环境，设置 OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1。
