#pragma once

#include <cstddef>
#include "snn/core/memory.hpp"
#include "snn/neuron/neuron.hpp"

namespace snn
{
    struct IFConfig
    {
        float v_threshold = 1.0f;
        float v_reset = 0.0f;
    };

    class IFNeuron : public Neuron
    {
    public:
        IFConfig cfg_;
        F32Buffer v_;

        explicit IFNeuron(std::size_t size, IFConfig cfg = {})
            : cfg_(cfg),
              v_(size, cfg.v_reset)
        {
        }

        void step(F32Buffer &buffer) override
        {
            float *SNN_RESTRICT buf_ptr = SNN_ASSUME_ALIGNED(buffer.data(), 64);
            float *SNN_RESTRICT v_ptr = SNN_ASSUME_ALIGNED(v_.data(), 64);

            const std::size_t len = v_.size();
            const float v_th = cfg_.v_threshold;
            const float v_reset = cfg_.v_reset;

#pragma omp simd aligned(buf_ptr, v_ptr : 64)
            for (std::size_t i = 0; i < len; ++i)
            {
                float v = v_ptr[i] + buf_ptr[i];

                bool f = v >= v_th;

                buf_ptr[i] = f ? 1.0f : 0.0f;
                v_ptr[i] = f ? v_reset : v;
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