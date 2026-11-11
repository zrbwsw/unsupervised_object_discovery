# VOC07_compare

从 VOC07 的 5011 张完整评测图片中选择前 50 张。按 gap_sum = (CondGen IoU − DINO IoU) + (CondGen IoU − iBOT IoU) + (CondGen IoU − MAE IoU) 降序；同分按 CondGen IoU 降序、原图文件名升序。
指标均取自已有 predictions.jsonl 的 max_iou（范围 0–1）；它是预测框与全部有效 GT 框的最大 IoU。单图 CorLoc 成功标记为 max IoU >= 0.5。

固定使用用户指定阈值：DINO=0.30，iBOT=0.15，MAE=0.35，CondGen=0.45；没有对每张图重新选择 tau。
compare 的 gap_sum 范围为 −3 到 3，gap_mean = gap_sum/3，另记录相对最强基线的 gap_min。
这是按 GT 指标筛选的定性示例，不是随机样本或全数据集性能估计；框 IoU 不等同于分割质量。本目录中有 0/50 张 CondGen 预测框覆盖整图，严格保留指标排名而不作外观过滤。

## 图像与协议

原图直接复制；prediction_bbox.png 绘制已有评测框（红色）。mask 由相同 checkpoint 在**原始分辨率**重新运行 TokenCut 得到，要求重算 bbox 与已保存 bbox 完全一致。
使用冻结 ViT-B/16 的最后一层 LayerNorm 后 output patch features、tau 如上、eps=1e-5、二值图，补齐右/下边到 16 的倍数，不使用 CRF、bilateral solver、形态学等后处理。
红色 mask 使用 seed 所在连通分量，按 patch_size 最近邻展开再裁剪；前景颜色为 0.3×原像素+0.7×红色。
本批次没有使用 1024 输入重推理，避免改变排名对应的检测结果；overview/contact_sheet 的缩放仅用于浏览。
GT 只用于排序、核对与单独蓝框展示，不输入模型或 mask 推理。

## 每个样本的材料

- original.jpg：原始 JPEG 文件。
- metrics.json：所有四方法 IoU、成功标记、差值、框、原始结果文件位置。
- ground_truth.json、annotation.xml/annotation.json、ground_truth_boxes.png：评分所用标注和蓝色 GT 框图。
- condgen/ 或四方法子目录：prediction_bbox.png、mask_overlay.png、mask.png（二值前景）、foreground_patch.npy、eigenvector.npy、rendering.json（重推理核对记录）、prediction.json、source_run_config.json。
- overview.jpg：原图和预测框/mask 拼图。

index.csv 保存本目录排名，all_scores.csv 保存全数据集排序，contact_sheet.jpg 便于浏览所选 50 张。来源详情见 sources.json。

| 方法 | tau | 全量 CorLoc (%) |
| --- | ---: | ---: |
| dino | 0.30 | 60.1876 |
| ibot | 0.15 | 43.4444 |
| mae | 0.35 | 31.9697 |
| condgen | 0.45 | 67.3319 |
