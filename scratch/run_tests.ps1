$env:PATH = "$env:USERPROFILE\mingw64\bin;" + $env:PATH
cd C:\Users\Varun\Downloads\HFT\build
cmake -G "MinGW Makefiles" -DHFT_BUILD_TESTS=ON .. 2>&1 | Out-File cmake_log.txt
mingw32-make hft_tests 2>&1 | Out-File make_log.txt
.\hft_tests.exe --gtest_filter=OrderManagerTest.* 2>&1 | Out-File test_log.txt
Get-Content test_log.txt
