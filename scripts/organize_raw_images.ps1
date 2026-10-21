# Organize raw images: flatten folders, rename sequentially
$rawDir = "A:\grain\backend\data\raw"
$folders = @("Rice grains - 100", "Rice grains - 100 2")

# Get all image files from subfolders
$allImages = @()
foreach ($folder in $folders) {
    $folderPath = Join-Path $rawDir $folder
    if (Test-Path $folderPath) {
        $images = Get-ChildItem -Path $folderPath -File | Sort-Object Name
        $allImages += $images
    }
}

Write-Host "Found $($allImages.Count) images total"

# Copy all images to raw directory
$index = 1
foreach ($img in $allImages) {
    $ext = $img.Extension
    $newName = "$index$ext"
    $destPath = Join-Path $rawDir $newName
    Copy-Item -Path $img.FullName -Destination $destPath
    Write-Host "Copied $($img.Name) -> $newName"
    $index++
}

# Remove the source folders
foreach ($folder in $folders) {
    $folderPath = Join-Path $rawDir $folder
    if (Test-Path $folderPath) {
        Remove-Item -Path $folderPath -Recurse -Force
        Write-Host "Removed folder: $folder"
    }
}

Write-Host "Done. All images renamed 1 to $($allImages.Count)"
