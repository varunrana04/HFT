#pragma once

#include <thread>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace hft {
namespace utils {

/**
 * @brief Pin the calling thread to a specific CPU core.
 * 
 * Crucial for avoiding OS context switching in the hot path.
 * In a production Linux environment, the target core should be isolated
 * using the `isolcpus` kernel parameter.
 * 
 * @param core_id The zero-indexed CPU core number to pin to.
 * @return true if successful, false otherwise.
 */
inline bool pin_thread_to_core(int core_id) {
#ifdef _WIN32
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    DWORD_PTR result = SetThreadAffinityMask(thread, mask);
    if (result == 0) {
        std::cerr << "[ERROR] Failed to pin thread to core " << core_id 
                  << " (Windows Error: " << GetLastError() << ")\n";
        return false;
    }
    return true;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        std::cerr << "[ERROR] Failed to pin thread to core " << core_id 
                  << " (pthread error: " << result << ")\n";
        return false;
    }
    return true;
#endif
}

/**
 * @brief Set the thread priority to the maximum realtime priority.
 * 
 * On Linux, this requires CAP_SYS_NICE privileges and uses SCHED_FIFO.
 */
inline void set_thread_realtime_priority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#else
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
#endif
}

} // namespace utils
} // namespace hft
