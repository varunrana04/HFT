#pragma once
/**
 * @file memory_pool.h
 * @brief Lock-free, pre-allocated memory pool (arena allocator).
 *
 * Eliminates dynamic allocation on the hot path. Objects are pre-allocated
 * at startup into a contiguous block, and recycled via a free-list when
 * released. This guarantees:
 *   - O(1) allocation and deallocation
 *   - Zero calls to malloc/free/new/delete during trading
 *   - Cache-friendly contiguous memory layout
 *   - No garbage collection pauses
 *
 * Thread safety: This pool is designed for SINGLE-THREAD use on the hot
 * path. If multi-thread access is needed, wrap with per-thread pools
 * (thread-local storage).
 *
 * @tparam T    The object type to pool (must be trivially copyable)
 * @tparam Size Maximum number of objects in the pool
 */

#include <cstddef>
#include <cstdint>
#include <array>
#include <cassert>
#include <new>      // std::launder
#include <type_traits>

namespace hft {

template<typename T, size_t Size>
class MemoryPool {
    static_assert(std::is_trivially_copyable_v<T>,
        "MemoryPool requires trivially copyable types to avoid destructor issues");
    static_assert(Size > 0, "Pool size must be positive");

public:
    MemoryPool() noexcept {
        // Initialize the free list: each slot points to the next
        for (size_t i = 0; i < Size - 1; ++i) {
            free_list_[i] = static_cast<uint32_t>(i + 1);
        }
        free_list_[Size - 1] = SENTINEL;
        head_ = 0;
        allocated_ = 0;
    }

    /**
     * @brief Allocate an object from the pool.
     * @return Pointer to an uninitialized T, or nullptr if pool exhausted
     */
    [[nodiscard]] T* allocate() noexcept {
        if (head_ == SENTINEL) {
            return nullptr; // Pool exhausted
        }

        uint32_t index = head_;
        head_ = free_list_[index];
        ++allocated_;

        return std::launder(reinterpret_cast<T*>(&storage_[index]));
    }

    /**
     * @brief Return an object to the pool for reuse.
     * @param ptr Pointer previously obtained from allocate()
     *
     * The caller is responsible for ensuring ptr was obtained from THIS pool.
     * Returning a foreign pointer is undefined behavior.
     */
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;

        // Calculate the index from the pointer offset
        auto* base = reinterpret_cast<uint8_t*>(&storage_[0]);
        auto* target = reinterpret_cast<uint8_t*>(ptr);
        size_t index = static_cast<size_t>(target - base) / sizeof(Storage);

        assert(index < Size && "Pointer does not belong to this pool");

        free_list_[index] = head_;
        head_ = static_cast<uint32_t>(index);
        --allocated_;
    }

    /// Number of objects currently allocated
    [[nodiscard]] size_t allocated_count() const noexcept { return allocated_; }

    /// Number of objects available for allocation
    [[nodiscard]] size_t available_count() const noexcept { return Size - allocated_; }

    /// Total capacity of the pool
    [[nodiscard]] constexpr size_t capacity() const noexcept { return Size; }

    /// Check if the pool is fully allocated
    [[nodiscard]] bool is_full() const noexcept { return head_ == SENTINEL; }

    /// Check if the pool is empty (all slots available)
    [[nodiscard]] bool is_empty() const noexcept { return allocated_ == 0; }

    /// Reset the pool — invalidates all outstanding pointers
    void reset() noexcept {
        for (size_t i = 0; i < Size - 1; ++i) {
            free_list_[i] = static_cast<uint32_t>(i + 1);
        }
        free_list_[Size - 1] = SENTINEL;
        head_ = 0;
        allocated_ = 0;
    }

private:
    static constexpr uint32_t SENTINEL = std::numeric_limits<uint32_t>::max();

    /// Storage block — aligned to T's alignment requirement
    using Storage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    std::array<Storage, Size>   storage_;     ///< Contiguous object storage
    std::array<uint32_t, Size>  free_list_;   ///< Intrusive free list
    uint32_t                    head_;        ///< Index of next free slot
    size_t                      allocated_;   ///< Count of live objects
};

} // namespace hft
