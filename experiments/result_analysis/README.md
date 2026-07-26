# Manuscript result figures

该目录只保存论文当前使用的结果图、图件完整性清单以及图—源数据映射。
所有面板级 CSV 的唯一权威副本位于
`experiments/manuscript_figure_source_data/`，本目录不再复制源数据。

## 目录结构

```text
result_analysis/
|-- figures/
|   `-- <figure_id>/
|       |-- <figure_id>.png
|       |-- <figure_id>.pdf
|       |-- <figure_id>.svg
|       `-- <figure_id>.tiff
|-- scripts/
|   `-- package_manuscript_results.py
|-- manifest.csv
|-- figure_source_map.csv
`-- README.md
```

`molecular_origin_analysis/native_exports/` 额外保存计算流程的原生导出及
17.8 cm 可读性检查图；它们在 `manifest.csv` 中标记为
`figure_auxiliary`，不是当前紧凑投稿图。

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

每个正式子目录中的 PNG 都是作者提供的论文投稿版本。TIFF 是从该 PNG
进行的无损 LZW、600 dpi 转换。存在完全匹配的原生 Matplotlib
PDF/SVG 时直接保留原生文件。格式来源记录在 `manifest.csv`。

## 图—源数据映射

`figure_source_map.csv` 记录面板编号、权威 CSV 路径、行列数、说明及
SHA-256。`csv_file` 必须指向
`experiments/manuscript_figure_source_data/`；如果发现
`result_analysis/source_data/`，说明旧版重复打包尚未清理。

## 重新整理

从项目根目录运行：

```powershell
python experiments\manuscript_figure_source_data\rebuild_manifest.py
python experiments\result_analysis\scripts\package_manuscript_results.py `
  --manuscript-figure-dir "C:\Users\user\Downloads\OD-TwoColumn (June 2026)\Fig"
```

结果打包脚本不会重新训练模型或改变统计结果，只整理正式图件、生成缺失
格式、验证权威 CSV，并重写图件清单及图—数据映射。
