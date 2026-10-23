#pragma once

#include <cstddef>
#include "snn/core/compiler.hpp"
#include "snn/core/memory.hpp"
#include "snn/synapse/synapse.hpp"

namespace snn
{
    class SparseSynapse : public Synapse
    {
    public:
        SynapseMode mode_;
        std::size_t pre_;
        std::size_t post_;
        IndexBuffer c_;
        IndexBuffer i_;
        F32Buffer w_;

        explicit SparseSynapse(SynapseMode mode, std::size_t pre, std::size_t post)
            : mode_(mode),
              pre_(pre),
              post_(post),
              c_(0),
              i_(0),
              w_(0)
        {
        }

        void step(const F32Buffer &spike_buffer, F32Buffer &buffer) override
        {
            float *SNN_RESTRICT buf_ptr = SNN_ASSUME_ALIGNED(buffer.data(), 64);
            const float *SNN_RESTRICT spike_ptr = SNN_ASSUME_ALIGNED(spike_buffer.data(), 64);
            const float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(w_.data(), 64);
            const std::size_t *SNN_RESTRICT c_ptr = SNN_ASSUME_ALIGNED(c_.data(), 64);
            const std::size_t *SNN_RESTRICT i_ptr = SNN_ASSUME_ALIGNED(i_.data(), 64);

            if (mode_ == SynapseMode::Front)
            {
                std::size_t offset = 0;
                const std::size_t i_len = pre_;

                for (std::size_t i = 0; i < i_len; ++i)
                {
                    if (spike_ptr[i] > 0.0f)
                    {
                        const std::size_t k_len = c_ptr[i];
                        const std::size_t *SNN_RESTRICT row_dst = i_ptr + offset;
                        const float *SNN_RESTRICT row_w = w_ptr + offset;

#pragma clang loop unroll_count(8)
                        for (std::size_t k = 0; k < k_len; ++k)
                        {
                            buf_ptr[row_dst[k]] += row_w[k];
                        }
                    }

                    offset += c_ptr[i];
                }
            }
            else
            {
                std::size_t offset = 0;
                const std::size_t j_len = post_;

                for (std::size_t j = 0; j < j_len; ++j)
                {
                    const std::size_t k_len = c_ptr[j];
                    const std::size_t *SNN_RESTRICT row_dst = i_ptr + offset;
                    const float *SNN_RESTRICT row_w = w_ptr + offset;

                    float sum = 0.0f;

#pragma clang loop unroll_count(8)
                    for (std::size_t k = 0; k < k_len; ++k)
                    {
                        sum += row_w[k] * spike_buffer[row_dst[k]];
                    }

                    buf_ptr[j] += sum;
                }
            }
        }
    };
}