$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$projectPython = Join-Path $repo "TFproject\Scripts\python.exe"
$condaPython = "D:\anaconda3\envs\TFproject\python.exe"
if (Test-Path $projectPython) {
    $python = $projectPython
} elseif (Test-Path $condaPython) {
    $python = $condaPython
} else {
    throw "Cannot find TFproject Python. Checked $projectPython and $condaPython"
}

$mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
$device = if ($args.Count -gt 1) { $args[1] } else { "cuda" }
$out = Join-Path $repo "diagnostics\framework_$mode"
New-Item -ItemType Directory -Force -Path $out | Out-Null

if ($mode -eq "smoke") {
    & $python experiments\synthetic_path_control.py `
        --samples_per_class 20 `
        --output (Join-Path $out "synthetic_path_control_smoke.csv") `
        --json_output (Join-Path $out "synthetic_path_control_smoke.json")

    & $python experiments\memory_pollution_stress.py `
        --steps 100 `
        --pollutions 0.1 0.2 `
        --output (Join-Path $out "memory_pollution_smoke.csv") `
        --json_output (Join-Path $out "memory_pollution_smoke.json")

    & $python experiments\efficiency_scaling.py `
        --device $device `
        --class_grid 2 4 `
        --length_grid 64 128 `
        --phase_grid 3 5 `
        --batch_size 2 `
        --warmup 1 `
        --repeats 1 `
        --output (Join-Path $out "efficiency_scaling_smoke.csv") `
        --json_output (Join-Path $out "efficiency_scaling_smoke.json")

    & $python benchmark_official_splits.py `
        --experiment smoke_full_veto `
        --datasets Handwriting `
        --seeds 42 `
        --epochs 1 `
        --batch_size 16 `
        --device $device `
        --output (Join-Path $out "benchmark_smoke_full_veto.csv") `
        --json_output (Join-Path $out "benchmark_smoke_full_veto.json")
} elseif ($mode -eq "commands") {
    & $python experiments\run_experiment_suite.py `
        --suite all `
        --python $python `
        --device $device `
        --epochs 100 `
        --seeds 42 43 44 45 46 `
        --output_dir $out
} elseif ($mode -eq "mechanism") {
    & $python experiments\run_experiment_suite.py `
        --suite mechanism `
        --python $python `
        --device $device `
        --output_dir $out `
        --run
} else {
    throw "Unknown mode '$mode'. Use smoke, commands, or mechanism."
}

Write-Host "Done. Outputs are in $out"
