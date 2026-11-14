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
        IndexBuffer o_;
        IndexBuffer i_;
        F32Buffer w_;

        IndexBuffer o_post_;
        IndexBuffer i_post_;
        IndexBuffer w_post_;

        explicit SparseSynapse(std::size_t pre, std::size_t post)
            : pre_(pre),
              post_(post),
              o_(pre + 1, 0),
              i_(0),
              w_(0.0f),
              o_post_(0),
              i_post_(0),
              w_post_(0)
        {
        }

        void step(const BitBuffer &spike, F32Buffer &current) override
        {
            float *SNN_RESTRICT cur_ptr = SNN_ASSUME_ALIGNED(current.data(), 64);
            const std::size_t *SNN_RESTRICT o_ptr = SNN_ASSUME_ALIGNED(o_.data(), 64);
            const std::size_t *SNN_RESTRICT i_ptr = SNN_ASSUME_ALIGNED(i_.data(), 64);
            const float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(w_.data(), 64);

            spike.for_each_true([&](const std::size_t index)
                                {
                const std::size_t start = o_ptr[index];
                const std::size_t end = o_ptr[index + 1];

                const std::size_t *SNN_RESTRICT row_i = i_ptr + start;
                const float *SNN_RESTRICT row_w = w_ptr + start;
                const std::size_t count = end - start;

#pragma clang loop unroll_count(8)
                for (std::size_t k = 0; k < count; ++k)
                {
                    cur_ptr[row_i[k]] += row_w[k];
                } });
        }

        void for_each_post_impl(std::size_t pre_id, SynapseCallback cb) override
        {
            const std::size_t start = o_[pre_id];
            const std::size_t end = o_[pre_id + 1];

            const std::size_t *row_i = i_.data() + start;

            for (std::size_t idx = 0; idx < (end - start); ++idx)
            {
                cb(row_i[idx], start + idx);
            }
        }

        void for_each_pre_impl(std::size_t post_id, SynapseCallback cb) override
        {
            const std::size_t start = o_post_[post_id];
            const std::size_t end = o_post_[post_id + 1];

            const std::size_t *row_i = i_post_.data() + start;
            const std::size_t *row_w = w_post_.data() + start;

            for (std::size_t idx = 0; idx < (end - start); ++idx)
            {
                cb(row_i[idx], row_w[idx]);
            }
        }

        void enable_csc()
        {
            if (!o_post_.empty())
                return;

            const std::size_t total_syn = w_.size();

            o_post_.assign(post_ + 1, 0);
            i_post_.resize(total_syn);
            w_post_.resize(total_syn);

            for (std::size_t k = 0; k < total_syn; ++k)
            {
                ++o_post_[i_[k] + 1];
            }

            for (std::size_t p = 0; p < post_; ++p)
            {
                o_post_[p + 1] += o_post_[p];
            }

            IndexBuffer current_offsets = o_post_;

            for (std::size_t pre_id = 0; pre_id < pre_; ++pre_id)
            {
                const std::size_t start = o_[pre_id];
                const std::size_t end = o_[pre_id + 1];

                for (std::size_t k = start; k < end; ++k)
                {
                    const std::size_t post_id = i_[k];
                    const std::size_t dest = current_offsets[post_id]++;

                    i_post_[dest] = pre_id;
                    w_post_[dest] = k;
                }
            }
        }

        void disable_csc()
        {
            o_post_.clear();
            o_post_.shrink_to_fit();
            i_post_.clear();
            i_post_.shrink_to_fit();
            w_post_.clear();
            w_post_.shrink_to_fit();
        }

        [[nodiscard]] std::size_t pre_size() const noexcept override { return pre_; }
        [[nodiscard]] std::size_t post_size() const noexcept override { return post_; }

        [[nodiscard]] F32Buffer &weights() noexcept override { return w_; }
        [[nodiscard]] const F32Buffer &weights() const noexcept override { return w_; }
    };
}