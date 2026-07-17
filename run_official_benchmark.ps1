$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$python = Join-Path $repo "TFproject\Scripts\python.exe"
$csv = Join-Path $repo "diagnostics\official_split_benchmark_10ep_latest.csv"
$json = Join-Path $repo "diagnostics\official_split_benchmark_10ep_latest.json"
$log = Join-Path $repo "diagnostics\official_split_benchmark_10ep_latest.log"
$datasets = @(
    "EthanolConcentration",
    "FaceDetection",
    "Handwriting",
    "Heartbeat",
    "HHAR",
    "JapaneseVowels",
    "PEMS-SF",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "UWaveGestureLibrary"
)

& $python benchmark_official_splits.py `
    --datasets $datasets `
    --epochs 10 `
    --batch_size 16 `
    --device cpu `
    --output $csv `
    --json_output $json `
    *> $log
