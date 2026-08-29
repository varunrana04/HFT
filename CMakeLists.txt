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
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -mno-xop -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -mavx2 -mfma -mno-xop -fsanitize=address,undefined")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -fexceptions)
    
    option(USE_AVX512 "Enable AVX-512 instructions" OFF)
    if(USE_AVX512)
        add_compile_options(-mavx512f)
        message(STATUS "AVX-512 explicitly enabled via USE_AVX512")
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

# ─── IXWebSocket (fetched BEFORE hft_core so the target exists at link time) ─
include(FetchContent)
set(USE_TLS ON CACHE BOOL "IXWebSocket use TLS" FORCE)
set(USE_OPEN_SSL ON CACHE BOOL "IXWebSocket use OpenSSL" FORCE)
set(USE_MBED_TLS OFF CACHE BOOL "IXWebSocket use MbedTLS" FORCE)
set(USE_ZLIB OFF CACHE BOOL "IXWebSocket use ZLIB" FORCE)
FetchContent_Declare(
    ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.4
)
FetchContent_MakeAvailable(ixwebsocket)
message(STATUS "[hft_core] IXWebSocket client: ENABLED (TLS ON)")

# ─── Core Engine Library ───────────────────────────────────────
# Check if simdjson.cpp exists (it will inside Docker)
if(EXISTS "/usr/local/include/simdjson/simdjson.cpp")
    set(SIMDJSON_SRC "/usr/local/include/simdjson/simdjson.cpp")
else()
    set(SIMDJSON_SRC "") # Fallback for local IDE parsing
endif()

find_package(OpenSSL REQUIRED)

add_library(hft_core STATIC
    cpp/core/market_data.cpp
    cpp/core/order_book.cpp
    cpp/core/feature_engine.cpp
    cpp/core/signal_combiner.cpp
    cpp/core/kill_switch.cpp
    cpp/core/order_manager.cpp
    cpp/core/data_validator.cpp
    cpp/core/strategy_engine.cpp
    cpp/net/dpdk_rx.cpp
    cpp/net/dpdk_tx.cpp
    cpp/gateway/delta_ws.cpp
    cpp/gateway/delta_rest.cpp
    ${SIMDJSON_SRC}
)

target_include_directories(hft_core PUBLIC
    cpp/core cpp/net cpp/gateway
    /usr/local/include
    ${OPENSSL_INCLUDE_DIR}
)
target_link_libraries(hft_core PUBLIC ixwebsocket ${OPENSSL_LIBRARIES})
set_target_properties(hft_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Find uWebSockets dependencies (optional)
find_library(USOCKETS_LIB uSockets /usr/local/lib)
if(USOCKETS_LIB)
    target_link_libraries(hft_core PUBLIC ${USOCKETS_LIB} z ssl crypto)
    message(STATUS "[hft_core] uWebSockets gateway: ENABLED")
endif()

if(HFT_USE_DPDK)
    target_compile_definitions(hft_core PUBLIC HFT_USE_DPDK)
    target_include_directories(hft_core PUBLIC ${DPDK_INCLUDE_DIRS})
    
    # Remove -include rte_config.h as it breaks <cmath> and <atomic> in C++20 when processed before standard headers
    string(REPLACE "-include rte_config.h" "" DPDK_CFLAGS_OTHER_STR "${DPDK_CFLAGS_OTHER}")
    string(REPLACE ";" " " DPDK_CFLAGS_OTHER_STR "${DPDK_CFLAGS_OTHER_STR}")
    separate_arguments(DPDK_CFLAGS_CLEAN UNIX_COMMAND "${DPDK_CFLAGS_OTHER_STR}")
    
    target_compile_options(hft_core PUBLIC ${DPDK_CFLAGS_CLEAN})
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
        target_compile_options(hft_engine PRIVATE -fexceptions)
        target_include_directories(hft_engine PRIVATE cpp/core cpp/gateway cpp)
        if(NOT MSVC)
            target_link_options(hft_engine PRIVATE -static-libgcc -static-libstdc++)
        endif()
        message(STATUS "Python ${Python3_VERSION} found — building Python bindings")

        # Copy MinGW runtime DLLs next to the .pyd so Python can load it
        # without needing mingw64/bin in PATH (Windows DLL search path issue).
        set(MINGW_BIN "C:/msys64/mingw64/bin")
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

    else()
        message(STATUS "Python3 not found — skipping Python bindings")
    endif()
endif()

# ─── Live Engine Executable ────────────────────────────────────
add_executable(hft_engine_live cpp/main.cpp)
target_link_libraries(hft_engine_live PRIVATE hft_core)

# ─── Tests ─────────────────────────────────────────────────────
option(HFT_BUILD_TESTS "Build unit tests" OFF)
if(HFT_BUILD_TESTS)
    enable_testing()
    add_executable(hft_tests
        cpp/tests/test_main.cpp
        cpp/tests/test_types.cpp
        cpp/tests/test_spsc_queue.cpp
        cpp/tests/test_memory_pool.cpp
        cpp/tests/test_order_book.cpp
        cpp/tests/test_data_validator.cpp
        cpp/tests/test_feature_engine.cpp
        cpp/tests/test_kill_switch.cpp

        cpp/tests/test_strategy_engine.cpp
        cpp/tests/test_order_manager.cpp
    )
    find_package(GTest REQUIRED)
    target_link_libraries(hft_tests PRIVATE hft_core GTest::gtest GTest::gtest_main)
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
    add_executable(hft_sim
        cpp/bench/institutional_simulator.cpp
    )
    target_link_libraries(hft_bench PRIVATE hft_core)
    target_include_directories(hft_bench PRIVATE cpp/core)
    target_link_libraries(hft_sim PRIVATE hft_core)
    target_include_directories(hft_sim PRIVATE cpp/core)
endif()

# ─── Build summary ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "═══ HFT Engine Build Configuration ═══")
message(STATUS "  C++ Standard   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type     : ${CMAKE_BUILD_TYPE}")

message(STATUS "  Python bindings: ${HFT_BUILD_PYTHON}")
message(STATUS "  Tests          : ${HFT_BUILD_TESTS}")
message(STATUS "  Benchmarks     : ${HFT_BUILD_BENCHMARKS}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
