#pragma once

#include <cstddef>
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

    using F32Buffer = std::vector<float, AlignedAllocator<float, 64>>;
    using F64Buffer = std::vector<double, AlignedAllocator<float,64>>;
}