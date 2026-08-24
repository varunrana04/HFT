# CMake generated Testfile for 
# Source directory: /app
# Build directory: /app/build2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[hft_tests]=] "/app/build2/hft_tests")
set_tests_properties([=[hft_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/app/CMakeLists.txt;228;add_test;/app/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
