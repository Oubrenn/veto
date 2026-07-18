$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$python = "D:\anaconda3\envs\TFproject\python.exe"
$data = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
$outDir = Join-Path $repo "diagnostics\main_table"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$datasets = @(
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "Epilepsy",
    "ERing",
    "FingerMovements",
    "HandMovementDirection",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports"
)

& $python experiments\official_benchmark.py `
    --experiment main_table_veto_missing_bs64_10ep `
    --data_path $data `
    --datasets $datasets `
    --seeds 42 `
    --epochs 10 `
    --batch_size 64 `
    --device cuda `
    --no_drop_last `
    --diagnostic_batches 0 `
    --output (Join-Path $outDir "veto_missing_bs64_10ep.csv") `
    --json_output (Join-Path $outDir "veto_missing_bs64_10ep.json")
