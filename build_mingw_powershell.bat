@echo off
echo Cleaning old build files...
if exist build (
    rmdir /s /q build
)
echo Building using MinGW Makefiles...
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo CMake config failed!
    pause
    exit /b 1
)

cmake --build build
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build successful!
pause
