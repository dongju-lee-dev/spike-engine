#pragma once

#include <cstddef>
#include "snn/core/compiler.hpp"
#include "snn/core/memory.hpp"
#include "snn/synapse/synapse.hpp"

namespace snn
{
    class DenseSynapse : public Synapse
    {
    public:
        SynapseMode mode_;
        std::size_t pre_;
        std::size_t post_;
        F32Buffer w_;

        explicit DenseSynapse(SynapseMode mode, std::size_t pre, std::size_t post)
            : mode_(mode),
              pre_(pre),
              post_(post),
              w_(pre * post, 0.0f)
        {
        }

        void step(const F32Buffer &spike_buffer, F32Buffer &buffer) override
        {
            float *SNN_RESTRICT buf_ptr = SNN_ASSUME_ALIGNED(buffer.data(), 64);
            const float *SNN_RESTRICT spike_ptr = SNN_ASSUME_ALIGNED(spike_buffer.data(), 64);
            const float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(w_.data(), 64);

            const std::size_t n_pre = pre_;
            const std::size_t n_post = post_;

            if (mode_ == SynapseMode::Front)
            {
                for (std::size_t i = 0; i < n_pre; ++i)
                {
                    if (spike_ptr[i] > 0.0f)
                    {
                        const float *SNN_RESTRICT row_w = w_ptr + (i * n_post);

#pragma omp simd aligned(buf_ptr : 64)
                        for (std::size_t j = 0; j < n_post; ++j)
                        {
                            buf_ptr[j] += row_w[j];
                        }
                    }
                }
            }
            else
            {
                for (std::size_t j = 0; j < n_post; ++j)
                {
                    const float *SNN_RESTRICT row_w = w_ptr + (j * n_pre);
                    float sum = 0.0f;

#pragma omp simd reduction(+ : sum) aligned(spike_ptr : 64)
                    for (std::size_t i = 0; i < n_pre; ++i)
                    {
                        sum += row_w[i] * spike_ptr[i];
                    }

                    buf_ptr[j] += sum;
                }
            }
        }

        [[nodiscard]] SynapseMode mode() const noexcept override { return mode_; }
        [[nodiscard]] std::size_t pre_size() const noexcept override { return pre_; }
        [[nodiscard]] std::size_t post_size() const noexcept override { return post_; }

        [[nodiscard]] F32Buffer &weights() noexcept override { return w_; }
        [[nodiscard]] const F32Buffer &weights() const noexcept override { return w_; }
    };
}