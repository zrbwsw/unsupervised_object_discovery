# Mask overlay display comparison

同一张原图和同一批保存的 patch masks，四个方法统一渲染三列：

1. 原始 nearest、alpha=0.70（已逐像素核对与已有 overlay 一致）。
2. bilinear soft alpha、alpha=0.70（只改变显示插值）。
3. bilinear soft alpha、alpha=0.45（降低红色强度）。

打开 comparison.jpg 查看对照。各 PNG 保持原图分辨率。
bilinear 先映射到补边尺寸再裁剪，连续值仅作为透明度，不阈值化或重算框。
这是显示插值示例；没有增加 mask 的真实空间信息，不是 DINOv3 1024 输入推理复现。
源 patch mask、二值 mask、预测框、指标和既有 overlay 均经哈希校验未改动。
若论文采用该显示方式，应注明所有方法统一使用 bilinear alpha interpolation for visualization only。
