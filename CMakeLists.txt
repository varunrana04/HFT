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

# ─── ONNX Runtime (optional) ──────────────────────────────────
# To enable full LightGBM ONNX inference in SignalCombiner:
#
#   cmake -DHFT_ONNX_SUPPORT=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
#
# Download pre-built onnxruntime from:
#   https://github.com/microsoft/onnxruntime/releases
# Extract and point ONNXRUNTIME_ROOT at the extracted directory.
# Example Windows path: C:/onnxruntime-win-x64-1.18.0
#
# When OFF (default), SignalCombiner compiles without any ONNX dependency
# and load_onnx_model() always returns false (graceful no-op).

option(HFT_ONNX_SUPPORT "Enable ONNX Runtime inference in SignalCombiner" OFF)

if(HFT_ONNX_SUPPORT)
    # Allow user to specify the root directory of a pre-built onnxruntime package
    set(ONNXRUNTIME_ROOT "" CACHE PATH
        "Root of extracted onnxruntime package (contains include/ and lib/)")

    # Try find_package first (works if onnxruntime is installed system-wide)
    find_package(onnxruntime QUIET
        HINTS "${ONNXRUNTIME_ROOT}"
              "${ONNXRUNTIME_ROOT}/lib/cmake/onnxruntime"
    )

    if(onnxruntime_FOUND)
        message(STATUS "Found onnxruntime via find_package")
        set(ORT_INCLUDE_DIRS "")
        set(ORT_LIBRARIES    onnxruntime::onnxruntime)

    elseif(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        # Manual path from extracted package
        set(ORT_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
        if(WIN32)
            file(GLOB ORT_LIB_CANDIDATES
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
                "${ONNXRUNTIME_ROOT}/lib/Release/onnxruntime.lib"
            )
        else()
            file(GLOB ORT_LIB_CANDIDATES
                "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
                "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.*"
                "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib"
            )
        endif()
        if(ORT_LIB_CANDIDATES)
            list(GET ORT_LIB_CANDIDATES 0 ORT_LIBRARIES)
            message(STATUS "onnxruntime found at: ${ONNXRUNTIME_ROOT}")
            message(STATUS "  Include: ${ORT_INCLUDE_DIRS}")
            message(STATUS "  Library: ${ORT_LIBRARIES}")
        else()
            message(FATAL_ERROR
                "HFT_ONNX_SUPPORT=ON but no onnxruntime library found under "
                "${ONNXRUNTIME_ROOT}/lib/. "
                "Download from https://github.com/microsoft/onnxruntime/releases "
                "and set -DONNXRUNTIME_ROOT=/path/to/extracted/dir")
        endif()

    else()
        message(FATAL_ERROR
            "HFT_ONNX_SUPPORT=ON but ONNXRUNTIME_ROOT is not set or invalid.\n"
            "Usage: cmake -DHFT_ONNX_SUPPORT=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..")
    endif()
endif()

# ─── DPDK Kernel Bypass (optional) ────────────────────────────
option(HFT_USE_DPDK "Enable DPDK for Kernel Bypass Networking" OFF)

if(HFT_USE_DPDK)
    # Using pkg-config is the recommended way to find libdpdk in newer DPDK versions
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK REQUIRED libdpdk)
    
    if(DPDK_FOUND)
        message(STATUS "Found DPDK: ${DPDK_VERSION}")
    else()
        message(FATAL_ERROR "HFT_USE_DPDK=ON but libdpdk not found via pkg-config.")
    endif()
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
    cpp/net/dpdk_rx.cpp
    cpp/net/dpdk_tx.cpp
)
target_include_directories(hft_core PUBLIC cpp/core)

if(HFT_ONNX_SUPPORT)
    target_compile_definitions(hft_core PUBLIC HFT_ONNX_SUPPORT)
    target_include_directories(hft_core SYSTEM PUBLIC ${ORT_INCLUDE_DIRS})
    target_link_libraries(hft_core PUBLIC ${ORT_LIBRARIES})
    message(STATUS "[hft_core] ONNX Runtime inference: ENABLED")
else()
    message(STATUS "[hft_core] ONNX Runtime inference: DISABLED "
                   "(build with -DHFT_ONNX_SUPPORT=ON to enable)")
endif()

if(HFT_USE_DPDK)
    target_compile_definitions(hft_core PUBLIC HFT_USE_DPDK)
    target_include_directories(hft_core SYSTEM PUBLIC ${DPDK_INCLUDE_DIRS})
    target_link_options(hft_core PUBLIC ${DPDK_LDFLAGS})
    target_link_libraries(hft_core PUBLIC ${DPDK_LIBRARIES})
    message(STATUS "[hft_core] DPDK Kernel Bypass: ENABLED")
else()
    message(STATUS "[hft_core] DPDK Kernel Bypass: DISABLED "
                   "(build with -DHFT_USE_DPDK=ON to enable)")
endif()

# ─── Python Bindings (optional, auto-fetches pybind11 if Python found) ──
option(HFT_BUILD_PYTHON "Build Python bindings" ON)
if(HFT_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)

        pybind11_add_module(hft_engine cpp/bindings/py_engine.cpp)
        target_link_libraries(hft_engine PRIVATE hft_core)
        target_include_directories(hft_engine PRIVATE cpp/core)
        message(STATUS "Python ${Python3_VERSION} found — building Python bindings")

        # Copy MinGW runtime DLLs next to the .pyd so Python can load it
        # without needing mingw64/bin in PATH (Windows DLL search path issue).
        set(MINGW_BIN "$ENV{USERPROFILE}/mingw64/bin")
        foreach(DLL
            "libgcc_s_seh-1.dll"
            "libstdc++-6.dll"
            "libwinpthread-1.dll")
            if(EXISTS "${MINGW_BIN}/${DLL}")
                add_custom_command(TARGET hft_engine POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${MINGW_BIN}/${DLL}"
                        "$<TARGET_FILE_DIR:hft_engine>/${DLL}"
                    COMMENT "Copying ${DLL} next to hft_engine.pyd"
                )
            endif()
        endforeach()

        # When ONNX is enabled, copy onnxruntime DLL next to the .pyd too
        if(HFT_ONNX_SUPPORT AND WIN32)
            file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll")
            foreach(DLL ${ORT_DLLS})
                add_custom_command(TARGET hft_engine POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${DLL}"
                        "$<TARGET_FILE_DIR:hft_engine>/"
                    COMMENT "Copying ${DLL} next to hft_engine.pyd"
                )
            endforeach()
        endif()

    else()
        message(STATUS "Python3 not found — skipping Python bindings")
    endif()
endif()

# ─── Tests ─────────────────────────────────────────────────────
option(HFT_BUILD_TESTS "Build unit tests" ON)
if(HFT_BUILD_TESTS)
    enable_testing()
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
        target_compile_options(hft_tests PRIVATE
            -Wno-unused-variable -Wno-unused-function)
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

# ─── Build summary ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "═══ HFT Engine Build Configuration ═══")
message(STATUS "  C++ Standard   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Support   : ${HFT_ONNX_SUPPORT}")
message(STATUS "  Python bindings: ${HFT_BUILD_PYTHON}")
message(STATUS "  Tests          : ${HFT_BUILD_TESTS}")
message(STATUS "  Benchmarks     : ${HFT_BUILD_BENCHMARKS}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
