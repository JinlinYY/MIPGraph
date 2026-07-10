@echo off
setlocal
set PYTHONUNBUFFERED=1
cd /d D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction
if not exist outputs\property_specialists_ilthermo_interpolated_seed42 mkdir outputs\property_specialists_ilthermo_interpolated_seed42

E:\anaconda\envs\ggnn39\python.exe scripts\run_property_specialists.py ^
  --config configs\default.yaml ^
  --checkpoint outputs\property_branch_sequence_decoupled_log_seed42\checkpoints\decoupled_log_branch_step06_ThermalConductivity_seed42\best_model.pt ^
  --clean-csv data\processed_ilthermo_interpolated\il_multiprop_clean.csv ^
  --arrays-path data\processed_ilthermo_interpolated\il_multiprop_arrays.npz ^
  --graph-cache data\processed_ilthermo_interpolated\graph_cache.pt ^
  --split-path data\processed_ilthermo_interpolated\splits\il_level_seed42.json ^
  --properties Density,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity,Viscosity ^
  --seeds 42 ^
  --split-seed 42 ^
  --output-root outputs\property_specialists_ilthermo_interpolated_seed42 ^
  --epochs 80 ^
  --lr 0.0001 ^
  --patience 20 ^
  --batch-size 512 ^
  --validate-every 4 ^
  --focus-weight 4.0 ^
  --background-weight 0.0 ^
  --freeze-mode property_branch ^
  --num-workers 0 ^
  --monitor-space log ^
  --disable-property-coupling ^
  --skip-existing ^
  1> outputs\property_specialists_ilthermo_interpolated_seed42\run_stdout.log ^
  2> outputs\property_specialists_ilthermo_interpolated_seed42\run_stderr.log

exit /b %errorlevel%
