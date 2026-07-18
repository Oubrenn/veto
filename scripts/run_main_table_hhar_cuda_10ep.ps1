$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
$env:DISABLE_TQDM = "1"

$python = "D:\anaconda3\envs\TFproject\python.exe"
$data = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
$outDir = Join-Path $repo "diagnostics\main_table_baselines_10ep"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$models = @(
    "hc2_lite",
    "rocket",
    "multirocket",
    "inceptiontime",
    "timesnet",
    "mts2graph",
    "simtsc",
    "tma_gat",
    "tapnet",
    "pdftime"
)

foreach ($model in $models) {
    & $python experiments\official_benchmark.py `
        --experiment "main_table_$model`_10ep" `
        --model $model `
        --data_path $data `
        --datasets "HHAR" `
        --seeds 42 `
        --epochs 10 `
        --batch_size 128 `
        --device cuda `
        --no_drop_last `
        --diagnostic_batches 0 `
        --output (Join-Path $outDir "$model`_HHAR.csv") `
        --json_output (Join-Path $outDir "$model`_HHAR.json") `
        > (Join-Path $outDir "$model`_HHAR.out.log") 2> (Join-Path $outDir "$model`_HHAR.err.log")

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$model on HHAR exited with code $LASTEXITCODE."
    }
}

& $python experiments\official_benchmark.py `
    --experiment "main_table_veto_10ep" `
    --model "veto" `
    --data_path $data `
    --datasets "HHAR" `
    --seeds 42 `
    --epochs 10 `
    --batch_size 64 `
    --device cuda `
    --no_drop_last `
    --diagnostic_batches 0 `
    --output (Join-Path $outDir "veto_HHAR.csv") `
    --json_output (Join-Path $outDir "veto_HHAR.json") `
    > (Join-Path $outDir "veto_HHAR.out.log") 2> (Join-Path $outDir "veto_HHAR.err.log")

if ($LASTEXITCODE -ne 0) {
    Write-Warning "veto on HHAR exited with code $LASTEXITCODE."
}
