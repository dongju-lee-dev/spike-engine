#pragma once

#include <cstddef>
#include "snn/core/memory.hpp"
#include "snn/neuron/neuron.hpp"

namespace snn
{
    struct LIFConfig
    {
        float v_threshold = 1.0f;
        float v_reset = 0.0f;
        float decay = 0.9f;
    };

    class LIFNeuron : public Neuron
    {
    public:
        LIFConfig cfg_;
        F32Buffer v_;

        explicit LIFNeuron(std::size_t size, LIFConfig cfg = {})
            : cfg_(cfg),
              v_(size, cfg.v_reset)
        {
        }

    private:
        [[nodiscard]] inline std::uint64_t step_tile(
            const std::size_t base_idx,
            const std::size_t count,
            const float *SNN_RESTRICT cur_ptr,
            float *SNN_RESTRICT v_ptr) const noexcept
        {
            std::uint64_t spike_word = 0ULL;
            const float v_th = cfg_.v_threshold;
            const float v_reset = cfg_.v_reset;
            const float decay = cfg_.decay;

#pragma omp simd aligned(v_ptr, cur_ptr : 64) reduction(| : spike_word)
            for (std::size_t b = 0; b < count; ++b)
            {
                const std::size_t idx = base_idx + b;
                float v = decay * v_ptr[idx] + cur_ptr[idx];
                bool f = (v >= v_th);

                spike_word |= (static_cast<std::uint64_t>(f) << b);
                v_ptr[idx] = f ? v_reset : v;
            }

            return spike_word;
        }

    public:
        void step(const F32Buffer &current, BitBuffer &spike) override
        {
            float *SNN_RESTRICT v_ptr = SNN_ASSUME_ALIGNED(v_.data(), 64);
            const float *SNN_RESTRICT cur_ptr = SNN_ASSUME_ALIGNED(current.data(), 64);

            const std::size_t len = v_.size();
            const std::size_t full_word = len / 64;
            const std::size_t remainder = len % 64;

            for (std::size_t w = 0; w < full_word; ++w)
            {
                spike.set_word(w, step_tile(w * 64, 64, cur_ptr, v_ptr));
            }

            if (remainder > 0)
            {
                spike.set_word(full_word, step_tile(full_word * 64, remainder, cur_ptr, v_ptr));
            }
        }

        void reset() override
        {
            std::fill(v_.begin(), v_.end(), cfg_.v_reset);
        }

        [[nodiscard]] std::size_t size() const noexcept override
        {
            return v_.size();
        }
    };
}