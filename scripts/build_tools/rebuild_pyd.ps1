cd C:\Users\Varun\Downloads\HFT

# 1. Clean the old build cache
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue

# 2. Re-run CMake with Visual Studio 2022 generator
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

# 3. Build the Python extension
cmake --build build --config Release --target hft_engine --parallel

# 4. Copy the newly built .pyd to the python directory
$pyd = Get-ChildItem build\Release -Filter "hft_engine*.pyd" -Recurse | Select-Object -First 1
if ($pyd) {
    Copy-Item $pyd.FullName "python\hft_engine.cp311-win_amd64.pyd" -Force
    Copy-Item $pyd.FullName "python\hft_engine.pyd" -Force
    Write-Host "SUCCESS: Python binding deployed" -ForegroundColor Green
} else {
    $pyd2 = Get-ChildItem build -Filter "hft_engine*.pyd" | Select-Object -First 1
    if ($pyd2) {
        Copy-Item $pyd2.FullName "python\hft_engine.cp311-win_amd64.pyd" -Force
        Copy-Item $pyd2.FullName "python\hft_engine.pyd" -Force
        Write-Host "SUCCESS: Python binding deployed from build/" -ForegroundColor Green
    }
}
