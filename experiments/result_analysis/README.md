# Manuscript result figures and source data

该目录集中保存论文当前使用的 10 张结果图及其可审计源数据。

## 目录结构

```text
result_analysis/
|-- figures/
|   `-- <figure_id>/
|       |-- <figure_id>.png
|       |-- <figure_id>.pdf
|       |-- <figure_id>.svg
|       `-- <figure_id>.tiff
|-- source_data/
|   `-- <figure_id>/
|       `-- *.csv
|-- scripts/
|   `-- package_manuscript_results.py
|-- manifest.csv
|-- figure_source_map.csv
`-- README.md
```

## 图件范围

1. `auditable_virtual_screening_validation`
2. `dataset_statistics`
3. `figureS_application_decision_stability`
4. `figureS_application_derived_metrics`
5. `interpretability`
6. `figureS_heat_capacity_size_control`
7. `Intro-method`
8. `performance_results`
9. `reference_cell_scenario_audited`
10. `molecular_origin_analysis`

每个子目录中的 PNG 都是作者提供的论文投稿版本。TIFF 是从该 PNG
进行的无损 LZW、600 dpi 转换。存在同版本原生 Matplotlib PDF/SVG
时直接保留原生文件。

以下情况需要特别说明：

- `Intro-method` 是非定量概念示意图，没有数值型作图数据；其 CSV
  仅记录面板内容和“不适用”的源数据状态。
- 当前紧凑版 `molecular_origin_analysis` 没有完全匹配的原生矢量输出，
  因而其 PDF/SVG 由精确投稿 PNG 派生。
- `interpretability` 是由两张栅格结果图拼接的投稿总图；其 SVG
  为嵌入精确 PNG 的容器，不应描述为原生矢量图。

上述格式来源均记录在 `manifest.csv`，不得把栅格派生 SVG 误报为
可编辑的原生矢量结果。

## CSV 与面板映射

`figure_source_map.csv` 逐项记录：

- 图件与面板编号；
- 对应 CSV；
- CSV 的原始项目路径；
- 行数、列数和 SHA-256；
- 数据内容说明。

这里保存的是绘图所需的处理后 source data，不替代原始 ILThermo
数据、模型检查点或完整计算输出。

## 重新整理

从项目根目录运行：

```powershell
python experiments\result_analysis\scripts\package_manuscript_results.py `
  --manuscript-figure-dir "C:\Users\user\Downloads\OD-TwoColumn (June 2026)\Fig"
```

脚本不会重新训练模型或改变统计结果，只复制正式图件、生成缺失格式、
整理现有 CSV，并写入完整性清单。项目内的CSV来源统一读取
`experiments/manuscript_figure_source_data/`；不依赖已从公开仓库删除的
`result_fig/`。
