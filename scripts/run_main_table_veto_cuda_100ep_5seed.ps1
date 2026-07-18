$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$python = "D:\anaconda3\envs\TFproject\python.exe"
$data = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
$outDir = Join-Path $repo "diagnostics\main_table_100ep_5seed"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$datasets = @(
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2"
)

foreach ($dataset in $datasets) {
    $batchSize = 64
    if ($dataset -eq "EigenWorms") {
        $batchSize = 8
    }
    if ($dataset -eq "FaceDetection") {
        $batchSize = 16
    }

    & $python experiments\official_benchmark.py `
        --experiment "main_table_veto_100ep_5seed" `
        --data_path $data `
        --datasets $dataset `
        --seeds 42 43 44 45 46 `
        --epochs 100 `
        --batch_size $batchSize `
        --device cuda `
        --no_drop_last `
        --diagnostic_batches 0 `
        --output (Join-Path $outDir "$dataset.csv") `
        --json_output (Join-Path $outDir "$dataset.json") `
        *> (Join-Path $outDir "$dataset.log")
}
