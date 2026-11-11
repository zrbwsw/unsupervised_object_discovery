# Output layout

Canonical full evaluations use:

```text
outputs/<dataset>/<method>/tau_<value>/
```

For example, `outputs/VOC07/condgen/tau_0.45/` contains the CondGen VOC07
evaluation at that threshold. Each experiment directory contains its log,
predictions, configuration, and structured result files. The evaluation
scripts and queues refuse to overwrite an existing directory.

Queue logs, manifests, copied launch scripts, and status files are archived
under `_queues/`; CPU smoke runs are under `_smoke/`; qualitative
visualizations remain under `visualizations/`. The current canonical tree has
90 complete full evaluations: 30 each for VOC07, VOC12, and COCO20k.

The complete old-to-new mapping is in `relocation_manifest.tsv`. Each
relocated `run_config.json` also records its original and current output
directory.
