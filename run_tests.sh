#!/bin/bash
export PATH=/mingw64/bin:$PATH
cd /c/Users/Varun/Downloads/HFT/build
echo "Running CMake..."
cmake -G "MinGW Makefiles" -DHFT_BUILD_TESTS=ON ..
echo "Running Make..."
mingw32-make hft_tests
if [ -f hft_tests.exe ]; then
    echo "Running Tests..."
    ./hft_tests --gtest_filter=OrderManagerTest.*
else
    echo "hft_tests.exe not found! Compilation failed!"
fi
echo ""
echo "ALL DONE!"
read -p "Press Enter to close..."