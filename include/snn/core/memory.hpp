#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>
#include "compiler.hpp"

namespace snn
{

    template <typename T, size_t Alignment = 64>
    class AlignedAllocator
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        AlignedAllocator() noexcept = default;

        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

        T *allocate(std::size_t n)
        {
            if (n == 0)
                return nullptr;

            size_t bytes = n * sizeof(T);
            void *ptr = nullptr;

#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || defined(__GLIBC__)
            ptr = std::aligned_alloc(Alignment, (bytes + Alignment - 1) & ~(Alignment - 1));
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
            if (posix_memalign(&ptr, Alignment, bytes) != 0)
            {
                ptr = nullptr;
            }
#else
            ptr = std::malloc(bytes);
#endif
            if (!ptr)
            {
                throw std::bad_alloc();
            }

            return static_cast<T *>(ptr);
        }

        void deallocate(T *ptr, [[maybe_unused]] std::size_t n) noexcept
        {
            if (ptr)
            {
                std::free(ptr);
            }
        }

        template <typename U>
        struct rebind
        {
            using other = AlignedAllocator<U, Alignment>;
        };

        template <typename U>
        bool operator==(const AlignedAllocator<U, Alignment> &) const noexcept { return true; }

        template <typename U>
        bool operator!=(const AlignedAllocator<U, Alignment> &) const noexcept { return false; }
    };

    using IndexBuffer = std::vector<std::size_t, AlignedAllocator<std::size_t, 64>>;

    using U8Buffer = std::vector<uint8_t, AlignedAllocator<uint8_t, 64>>;
    using U16Buffer = std::vector<uint16_t, AlignedAllocator<uint16_t, 64>>;
    using U32Buffer = std::vector<uint32_t, AlignedAllocator<uint32_t, 64>>;
    using U64Buffer = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

    using I8Buffer = std::vector<int8_t, AlignedAllocator<int8_t, 64>>;
    using I16Buffer = std::vector<int16_t, AlignedAllocator<int16_t, 64>>;
    using I32Buffer = std::vector<int32_t, AlignedAllocator<int32_t, 64>>;
    using I64Buffer = std::vector<int64_t, AlignedAllocator<int64_t, 64>>;

    using F32Buffer = std::vector<float, AlignedAllocator<float, 64>>;
    using F64Buffer = std::vector<double, AlignedAllocator<double, 64>>;
}