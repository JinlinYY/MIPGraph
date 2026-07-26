# Molecular-origin analysis source data

该数据包保存分子结构—宏观性质分析的权威面板级 CSV，不保存结果图。

## 面板源数据

- `source_data/figure_main_molecular_origin_analysis_final_panel_a_source_data.csv`
- `source_data/figure_main_molecular_origin_analysis_final_panel_b_source_data.csv`
- `source_data/figure_main_molecular_origin_analysis_final_panel_c_source_data.csv`
- `source_data/figure_main_molecular_origin_analysis_final_panel_d_source_data.csv`
- `source_data/figure_si_heat_capacity_size_control_source_data.csv`

`column_dictionary.csv` 给出字段的化学和统计含义。关联、注意力对比和
匹配取代结果均为非因果证据；标记为 `exploratory=True` 的热导率结果
保留探索性限定。

对应正式图位于：

- `experiments/result_analysis/figures/molecular_origin_analysis/`
- `experiments/result_analysis/figures/figureS_heat_capacity_size_control/`

重新生成：

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage all
python experiments\molecular_origin_analysis\package_source_data.py
python experiments\manuscript_figure_source_data\rebuild_manifest.py
```
