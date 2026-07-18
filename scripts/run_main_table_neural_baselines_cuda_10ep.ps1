$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
$env:DISABLE_TQDM = "1"

$python = "D:\anaconda3\envs\TFproject\python.exe"
$data = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
$outDir = Join-Path $repo "diagnostics\main_table_baselines_10ep"
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

$models = @("timesnet", "tapnet", "mts2graph", "simtsc", "tma_gat")

foreach ($model in $models) {
    foreach ($dataset in $datasets) {
        $batchSize = 64
        if ($dataset -eq "EigenWorms") {
            $batchSize = 8
        }
        if ($dataset -eq "FaceDetection" -or $dataset -eq "PhonemeSpectra") {
            $batchSize = 32
        }

        $stdoutLog = Join-Path $outDir "$model`_$dataset.out.log"
        $stderrLog = Join-Path $outDir "$model`_$dataset.err.log"

        & $python experiments\official_benchmark.py `
            --experiment "main_table_$model`_10ep" `
            --model $model `
            --data_path $data `
            --datasets $dataset `
            --seeds 42 `
            --epochs 10 `
            --batch_size $batchSize `
            --device cuda `
            --no_drop_last `
            --diagnostic_batches 0 `
            --output (Join-Path $outDir "$model`_$dataset.csv") `
            --json_output (Join-Path $outDir "$model`_$dataset.json") `
            > $stdoutLog 2> $stderrLog

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$model on $dataset exited with code $LASTEXITCODE. See $stdoutLog and $stderrLog"
        }
    }
}
