cmake_minimum_required(VERSION 3.20)
project(hft_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Compiler Flags ────────────────────────────────────────────
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /fsanitize=address")
    add_compile_options(/W4 /permissive-)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

# ─── Core Engine Library ───────────────────────────────────────
add_library(hft_core STATIC
    cpp/core/market_data.cpp
    cpp/core/order_book.cpp
    cpp/core/features.cpp
    cpp/core/signal_combiner.cpp
    cpp/core/risk_manager.cpp
    cpp/core/order_manager.cpp
    cpp/core/data_validator.cpp
    cpp/core/strategy_engine.cpp
)
target_include_directories(hft_core PUBLIC cpp/core)

# ─── Python Bindings (Requires pybind11) ─────────────
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(hft_engine cpp/bindings/py_engine.cpp)
target_link_libraries(hft_engine PRIVATE hft_core)
message(STATUS "pybind11 fetched — Python bindings will be built")

# ─── Tests ─────────────────────────────────────────────────────
option(HFT_BUILD_TESTS "Build unit tests" ON)
if(HFT_BUILD_TESTS)
    enable_testing()
    
    # We bundle a minimal test harness (no external dependency needed)
    add_executable(hft_tests
        cpp/tests/test_main.cpp
        cpp/tests/test_types.cpp
        cpp/tests/test_spsc_queue.cpp
        cpp/tests/test_memory_pool.cpp
        cpp/tests/test_order_book.cpp
        cpp/tests/test_data_validator.cpp
        cpp/tests/test_features.cpp
        cpp/tests/test_risk_manager.cpp
        cpp/tests/test_strategy_engine.cpp
    )
    target_link_libraries(hft_tests PRIVATE hft_core)
    target_include_directories(hft_tests PRIVATE cpp/core cpp/tests)
    if(NOT MSVC)
        target_compile_options(hft_tests PRIVATE -Wno-unused-variable -Wno-unused-function)
    endif()
    add_test(NAME hft_tests COMMAND hft_tests)
endif()

# ─── Benchmarks ────────────────────────────────────────────────
option(HFT_BUILD_BENCHMARKS "Build latency benchmarks" ON)
if(HFT_BUILD_BENCHMARKS)
    add_executable(hft_bench
        cpp/bench/latency_profiler.cpp
    )
    target_link_libraries(hft_bench PRIVATE hft_core)
    target_include_directories(hft_bench PRIVATE cpp/core)
endif()
