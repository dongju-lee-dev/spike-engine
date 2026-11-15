#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
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

    using IndexBuffer = std::vector<std::size_t, AlignedAllocator<std::size_t, 64>>;

    using U8Buffer = std::vector<std::uint8_t, AlignedAllocator<std::uint8_t, 64>>;
    using U16Buffer = std::vector<std::uint16_t, AlignedAllocator<std::uint16_t, 64>>;
    using U32Buffer = std::vector<std::uint32_t, AlignedAllocator<std::uint32_t, 64>>;
    using U64Buffer = std::vector<std::uint64_t, AlignedAllocator<std::uint64_t, 64>>;

    using F32Buffer = std::vector<float, AlignedAllocator<float, 64>>;
    using F64Buffer = std::vector<double, AlignedAllocator<double, 64>>;

    class BitBuffer
    {
    public:
        std::vector<std::uint64_t, AlignedAllocator<std::uint64_t, 64>> data_;
        std::size_t size_;

        explicit BitBuffer()
            : data_(0),
              size_(0)
        {
        }

        explicit BitBuffer(std::size_t size, bool init)
            : data_((size + 63) / 64, init ? ~0ULL : 0ULL),
              size_(size)
        {
        }

        [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        [[nodiscard]] std::size_t size_word() const noexcept { return data_.size(); }

        [[nodiscard]] std::uint64_t get_word(std::size_t index) const noexcept { return data_[index]; }
        void set_word(std::size_t index, std::uint64_t value) noexcept { data_[index] = value; }

        void clear() noexcept
        {
            std::fill(data_.begin(), data_.end(), 0ULL);
        }

        void copy_from(const BitBuffer &src)
        {
            if (this == &src)
                return;

            if (size_ != src.size_)
            {
                size_ = src.size_;
                data_.resize(src.data_.size());
            }

            std::copy(src.data_.begin(), src.data_.end(), data_.begin());
        }

        void copy_to(BitBuffer &dest) const
        {
            dest.copy_from(*this);
        }

        template <typename F>
        inline void for_each_true(F &&func) const
        {
            const std::size_t num_words = data_.size();
            const std::uint64_t *words = data_.data();

            for (std::size_t w = 0; w < num_words; ++w)
            {
                std::uint64_t word = words[w];
                if (word == 0ULL)
                    continue;

                const std::size_t base_idx = w * 64;
                while (word != 0ULL)
                {
                    const std::uint64_t b = std::countr_zero(word);
                    func(base_idx + b);
                    word &= (word - 1ULL);
                }
            }
        }
    };
}