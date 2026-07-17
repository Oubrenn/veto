$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$python = Join-Path $repo "TFproject\Scripts\python.exe"
$diag = Join-Path $repo "diagnostics"

& $python benchmark_official_splits.py `
    --datasets EthanolConcentration SelfRegulationSCP2 `
    --epochs 10 `
    --batch_size 16 `
    --device cpu `
    --no_normalize `
    --output (Join-Path $diag "opt_ethanol_scp2_no_normalize_10ep.csv") `
    --json_output (Join-Path $diag "opt_ethanol_scp2_no_normalize_10ep.json")

& $python benchmark_official_splits.py `
    --datasets Handwriting `
    --epochs 20 `
    --batch_size 16 `
    --device cpu `
    --no_drop_last `
    --no_cf `
    --output (Join-Path $diag "opt_handwriting_20ep_no_drop_last_no_cf.csv") `
    --json_output (Join-Path $diag "opt_handwriting_20ep_no_drop_last_no_cf.json")

& $python benchmark_official_splits.py `
    --datasets FaceDetection `
    --epochs 10 `
    --batch_size 16 `
    --device cpu `
    --no_normalize `
    --no_cf `
    --output (Join-Path $diag "opt_facedetection_no_normalize_nocf_10ep.csv") `
    --json_output (Join-Path $diag "opt_facedetection_no_normalize_nocf_10ep.json")
