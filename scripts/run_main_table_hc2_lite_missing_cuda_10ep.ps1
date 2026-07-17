$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
$env:DISABLE_TQDM = "1"

$python = "D:\anaconda3\envs\TFproject\python.exe"
$data = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
$outDir = Join-Path $repo "diagnostics\main_table_baselines_10ep"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$datasets = @(
    "CharacterTrajectories",
    "EigenWorms",
    "FaceDetection",
    "JapaneseVowels",
    "MotorImagery",
    "PenDigits",
    "PhonemeSpectra"
)

foreach ($dataset in $datasets) {
    $batchSize = 64
    if ($dataset -eq "EigenWorms") {
        $batchSize = 4
    }
    if ($dataset -eq "FaceDetection" -or $dataset -eq "PhonemeSpectra") {
        $batchSize = 32
    }

    & $python benchmark_official_splits.py `
        --experiment "main_table_hc2_lite_10ep" `
        --model "hc2_lite" `
        --data_path $data `
        --datasets $dataset `
        --seeds 42 `
        --epochs 10 `
        --batch_size $batchSize `
        --device cuda `
        --no_drop_last `
        --diagnostic_batches 0 `
        --output (Join-Path $outDir "hc2_lite_$dataset.csv") `
        --json_output (Join-Path $outDir "hc2_lite_$dataset.json") `
        > (Join-Path $outDir "hc2_lite_$dataset.out.log") 2> (Join-Path $outDir "hc2_lite_$dataset.err.log")

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "hc2_lite on $dataset exited with code $LASTEXITCODE."
    }
}
