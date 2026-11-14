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
        std::size_t pre_;
        std::size_t post_;
        F32Buffer w_;

        explicit DenseSynapse(std::size_t pre, std::size_t post)
            : pre_(pre),
              post_(post),
              w_(pre * post, 0.0f)
        {
        }

        void step(const BitBuffer &spike, F32Buffer &current) override
        {
            float *SNN_RESTRICT cur_ptr = SNN_ASSUME_ALIGNED(current.data(), 64);
            const float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(w_.data(), 64);

            const std::size_t n_post = post_;

            spike.for_each_true([&](const std::size_t index)
                                {
                const float *SNN_RESTRICT row_w = index * n_post + w_ptr;

#pragma clang loop vectorize(enable)
                for (std::size_t i = 0; i < n_post; ++i)
                {
                    cur_ptr[i] += row_w[i];
                } });
        }

        void for_each_post_impl(std::size_t pre_id, SynapseCallback cb) override
        {
            const std::size_t base_w_idx = pre_id * post_;
            const std::size_t n_post = post_;

            for (std::size_t post_id = 0; post_id < n_post; ++post_id)
            {
                cb(post_id, base_w_idx + post_id);
            }
        }

        void for_each_pre_impl(std::size_t post_id, SynapseCallback cb) override
        {
            const std::size_t n_pre = pre_;
            const std::size_t stride = post_;

            for (std::size_t pre_id = 0; pre_id < n_pre; ++pre_id)
            {
                cb(pre_id, pre_id * stride + post_id);
            }
        }

        [[nodiscard]] std::size_t pre_size() const noexcept override { return pre_; }
        [[nodiscard]] std::size_t post_size() const noexcept override { return post_; }

        [[nodiscard]] F32Buffer &weights() noexcept override { return w_; }
        [[nodiscard]] const F32Buffer &weights() const noexcept override { return w_; }
    };
}