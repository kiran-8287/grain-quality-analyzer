# Flatten the raw image sub-folders into data/raw, rename 1..N, then remove the folders.
#
#   powershell -ExecutionPolicy Bypass -File scripts/organize_raw_images.ps1
#
# Writes a rename manifest (old name -> new name) so the original file names stay
# recoverable after the numeric rename.

param(
    [string]$RawDir = "A:\grain\backend\data\raw",
    [string[]]$Folders = @("Rice grains - 100", "Rice grains - 100 2"),
    [string]$ManifestPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RawDir)) {
    throw "Raw dir not found: $RawDir"
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path (Split-Path $RawDir -Parent) "work\raw_rename_map.csv"
}

# Collect every file from each source folder, preserving folder order then name order.
$allImages = @()
foreach ($folder in $Folders) {
    $folderPath = Join-Path $RawDir $folder
    if (-not (Test-Path -LiteralPath $folderPath)) {
        Write-Host "Skipping missing folder: $folder"
        continue
    }
    $images = @(Get-ChildItem -LiteralPath $folderPath -File | Sort-Object Name)
    Write-Host "Found $($images.Count) file(s) in '$folder'"
    foreach ($img in $images) {
        $allImages += [PSCustomObject]@{
            File    = $img
            Source  = $folder
        }
    }
}

if ($allImages.Count -eq 0) {
    throw "No files found in any source folder under $RawDir"
}

Write-Host "Copying $($allImages.Count) file(s) to $RawDir"

$manifest = @()
$index = 1
foreach ($entry in $allImages) {
    $img = $entry.File
    $newName = "$index$($img.Extension.ToLower())"
    $destPath = Join-Path $RawDir $newName
    Copy-Item -LiteralPath $img.FullName -Destination $destPath
    $manifest += [PSCustomObject]@{
        new_name = $newName
        old_name = $img.Name
        old_folder = $entry.Source
    }
    $index++
}

# Remove the now-empty source folders.
foreach ($folder in $Folders) {
    $folderPath = Join-Path $RawDir $folder
    if (Test-Path -LiteralPath $folderPath) {
        Remove-Item -LiteralPath $folderPath -Recurse -Force
        Write-Host "Removed folder: $folder"
    }
}

# Persist the rename manifest.
$manifestDir = Split-Path $ManifestPath -Parent
if (-not (Test-Path -LiteralPath $manifestDir)) {
    New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
}
$manifest | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8
Write-Host "Wrote rename manifest: $ManifestPath"

Write-Host "Done. $index file(s) total, renamed 1..$($allImages.Count)"
