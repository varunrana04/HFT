set PATH=%USERPROFILE%\mingw64\bin;%PATH%
cd C:\Users\Varun\Downloads\HFT\build
cmake -G "MinGW Makefiles" -DHFT_BUILD_TESTS=ON .. > cmake_log.txt 2>&1
mingw32-make hft_tests > make_log.txt 2>&1
.\hft_tests.exe --gtest_filter=OrderManagerTest.* > test_log.txt 2>&1
type test_log.txt
