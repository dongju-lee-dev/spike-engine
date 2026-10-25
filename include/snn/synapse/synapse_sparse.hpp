#pragma once

#include <cstddef>
#include "snn/core/memory.hpp"
#include "snn/synapse/synapse.hpp"

namespace snn
{
    class SparseSynapse : public Synapse
    {
    public:
        std::size_t pre_;
        std::size_t post_;
        IndexBuffer c_;
        IndexBuffer i_;
        F32Buffer w_;

        explicit SparseSynapse(std::size_t pre, std::size_t post)
            : pre_(pre),
              post_(post),
              c_(0),
              i_(0),
              w_(0)
        {
        }

        void step(const BitBuffer &spike, F32Buffer &current) override
        {
            float *SNN_RESTRICT cur_ptr = SNN_ASSUME_ALIGNED(current.data(), 64);
            const std::size_t *SNN_RESTRICT c_ptr = SNN_ASSUME_ALIGNED(c_.data(), 64);
            const std::size_t *SNN_RESTRICT i_ptr = SNN_ASSUME_ALIGNED(i_.data(), 64);
            const float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(w_.data(), 64);

            std::size_t count = 0;
            std::size_t ltidx = 0;

            spike.for_each_true([&](const std::size_t index)
                                {
                for (; ltidx < index; ++ltidx){
                    count += c_ptr[ltidx];
                }

                const std::size_t c = c_ptr[ltidx];
                const std::size_t *SNN_RESTRICT row_i = count + i_ptr;
                const float *SNN_RESTRICT row_w = count + w_ptr;

#pragma clang loop unroll_count(8)
                for (std::size_t i = 0; i < c; ++i)
                {
                    cur_ptr[row_i[i]] += row_w[i];
                } });
        }

        [[nodiscard]] std::size_t pre_size() const noexcept override { return pre_; }
        [[nodiscard]] std::size_t post_size() const noexcept override { return post_; }

        [[nodiscard]] F32Buffer &weights() noexcept override { return w_; }
        [[nodiscard]] const F32Buffer &weights() const noexcept override { return w_; }
    };
}