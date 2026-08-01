#pragma once
/**
 * @file spsc_queue.h
 * @brief Lock-free Single-Producer Single-Consumer (SPSC) ring buffer.
 *
 * This is the backbone of inter-thread and C++ ↔ Python communication.
 * Designed for the hot path with these guarantees:
 *   - Lock-free: uses std::atomic with acquire/release semantics
 *   - Cache-friendly: producer and consumer indices on separate cache lines
 *   - Zero allocation: fixed capacity, pre-allocated at construction
 *   - Power-of-2 capacity: enables fast modulo via bitwise AND
 *
 * Based on the Rigtorp SPSCQueue pattern, adapted for HFT use.
 *
 * @tparam T        Element type (must be trivially copyable)
 * @tparam Capacity Queue capacity (must be power of 2)
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <cassert>

#include "types.h"

namespace hft {

template<typename T, size_t Capacity = DEFAULT_QUEUE_CAPACITY>
class SPSCQueue {
    static_assert(std::is_trivially_copyable_v<T>,
        "SPSCQueue requires trivially copyable types for lock-free safety");
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
        "Capacity must be a positive power of 2");

public:
    SPSCQueue() noexcept : head_(0), tail_(0) {}

    /// Non-copyable, non-movable (contains atomics)
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    /**
     * @brief Try to push an item into the queue (producer side).
     *
     * @param item The item to enqueue
     * @return true if the item was successfully enqueued, false if queue is full
     *
     * This function is wait-free: it completes in constant time regardless
     * of what the consumer thread is doing.
     *
     * Memory ordering:
     *   - Load tail_ with relaxed (only producer writes head_)
     *   - Store head_ with release (ensures item is visible before head advances)
     */
    [[nodiscard]] bool try_push(const T& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & MASK;

        // Check if queue is full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }

        buffer_[head] = item;

        // Release: ensures the item write is visible before head advances
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to pop an item from the queue (consumer side).
     *
     * @param[out] item The dequeued item (only valid if return is true)
     * @return true if an item was successfully dequeued, false if queue is empty
     *
     * Memory ordering:
     *   - Load head_ with acquire (sees the latest item written by producer)
     *   - Store tail_ with release (frees the slot for the producer)
     */
    [[nodiscard]] bool try_pop(T& item) noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);

        // Check if queue is empty
        if (tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }

        item = buffer_[tail];

        // Release: frees the slot for the producer
        const size_t next_tail = (tail + 1) & MASK;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    /**
     * @brief Peek at the front item without removing it (consumer side).
     *
     * @param[out] item The front item (only valid if return is true)
     * @return true if the queue is non-empty, false otherwise
     */
    [[nodiscard]] bool try_peek(T& item) const noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        item = buffer_[tail];
        return true;
    }

    /// Current number of items in the queue (approximate, not synchronized)
    [[nodiscard]] size_t size_approx() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (head - tail) & MASK;
    }

    /// Check if the queue appears empty (approximate)
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed)
            == tail_.load(std::memory_order_relaxed);
    }

    /// Check if the queue appears full (approximate)
    [[nodiscard]] bool full() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & MASK;
        return next == tail_.load(std::memory_order_relaxed);
    }

    /// Maximum number of elements the queue can hold
    [[nodiscard]] constexpr size_t capacity() const noexcept {
        // One slot is always wasted to distinguish full from empty
        return Capacity - 1;
    }

private:
    static constexpr size_t MASK = Capacity - 1;

    // ── Cache-line isolation ──────────────────────────────────
    // Producer index and consumer index are on SEPARATE cache lines
    // to prevent false sharing between the producer and consumer threads.

    /// Producer writes head_, consumer reads head_
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;

    /// Consumer writes tail_, producer reads tail_
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;

    /// Ring buffer storage — contiguous for cache friendliness
    alignas(CACHE_LINE_SIZE) T buffer_[Capacity];
};

} // namespace hft
