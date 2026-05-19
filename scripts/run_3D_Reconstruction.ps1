<#
run_colmap_sparse_reconstruction.ps1
#>

param(
    [string]$WorkDir = "data\processed\red_grapes_v2\3D",
    [string]$ImageSubdir = "images",
    [string]$Colmap = "C:\tools\colmap-x64-windows-cuda\COLMAP.bat",
    [string]$CameraModel = "SIMPLE_RADIAL",
    [string]$CameraParamsFile = "",
    [string]$CameraParams = "",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path

if ([System.IO.Path]::IsPathRooted($WorkDir)) {
    $Work = $WorkDir
}
else {
    $Work = Join-Path $RepoRoot $WorkDir
}

$Images = Join-Path $Work $ImageSubdir
$Database = Join-Path $Work "database.db"
$Sparse = Join-Path $Work "sparse"
$Model = Join-Path $Sparse "0"
$SparsePly = Join-Path $Sparse "cluster_full_sparse.ply"

if (!(Test-Path $Colmap)) {
    throw "COLMAP not found: $Colmap"
}

if (!(Test-Path $Images)) {
    throw "Image folder not found: $Images"
}

if ($CameraParamsFile -ne "") {
    if ([System.IO.Path]::IsPathRooted($CameraParamsFile)) {
        $CameraFile = $CameraParamsFile
    }
    else {
        $CameraFile = Join-Path $RepoRoot $CameraParamsFile
    }

    if (!(Test-Path $CameraFile)) {
        throw "Camera parameter file not found: $CameraFile"
    }

    $CameraParams = Get-Content $CameraFile |
        Where-Object { $_.Trim() -ne "" -and !$_.Trim().StartsWith("#") } |
        Select-Object -First 1
}

if ($Overwrite) {
    Remove-Item $Database -Force -ErrorAction SilentlyContinue
    Remove-Item $Sparse -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $Sparse -Force | Out-Null

$FeatureArgs = @(
    "feature_extractor",
    "--database_path", $Database,
    "--image_path", $Images,
    "--ImageReader.single_camera", "1",
    "--ImageReader.camera_model", $CameraModel,
    "--FeatureExtraction.type", "SIFT",
    "--FeatureExtraction.use_gpu", "1",
    "--FeatureExtraction.max_image_size", "2400"
)

if ($CameraParams -ne "") {
    $FeatureArgs += @("--ImageReader.camera_params", $CameraParams)
}

& $Colmap @FeatureArgs
if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed." }

& $Colmap exhaustive_matcher `
    --database_path $Database `
    --FeatureMatching.use_gpu 1 `
    --FeatureMatching.guided_matching 1

if ($LASTEXITCODE -ne 0) { throw "Feature matching failed." }

& $Colmap mapper `
    --database_path $Database `
    --image_path $Images `
    --output_path $Sparse `
    --Mapper.min_num_matches 15 `
    --Mapper.init_min_num_inliers 30 `
    --Mapper.abs_pose_min_num_inliers 15 `
    --Mapper.ba_refine_focal_length 1 `
    --Mapper.ba_refine_principal_point 0 `
    --Mapper.ba_refine_extra_params 0

if ($LASTEXITCODE -ne 0) { throw "Sparse reconstruction failed." }

if (!(Test-Path (Join-Path $Model "points3D.bin"))) {
    throw "Sparse model was not created at: $Model"
}

& $Colmap model_converter `
    --input_path $Model `
    --output_path $SparsePly `
    --output_type PLY

if ($LASTEXITCODE -ne 0) { throw "PLY export failed." }
