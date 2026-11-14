#pragma once

#include <algorithm>

#include "snn/learning/learning.hpp"
#include "snn/neuron/neuron.hpp"
#include "snn/synapse/synapse.hpp"
#include "snn/synapse/synapse_utils.hpp"

namespace snn
{
    enum class STDPKernel
    {
        Pair,
        Triplet,
    };

    enum class STDPWeight
    {
        Add,
        Multiply,
    };

    enum class STDPMode
    {
        Asymmetric,
        Symmetric,
    };

    template <STDPKernel kernel, STDPWeight weight, STDPMode mode>
    class STDP : public LearningRule
    {
    public:
        float a_plus_;
        float a_minus_;
        float t_plus_;
        float t_minus_;
        float w_min_;
        float w_max_;

        template <typename = std::enable_if_t<kernel == STDPKernel::Pair>>
        explicit STDP()
        {
        }

        template <typename = std::enable_if_t<kernel == STDPKernel::Triplet>>
        explicit STDP()
        {
        }

        void start(const std::any &) override
        {
        }

        void step(const std::size_t tick) override
        {
        }

        void end(const std::any &) override
        {
        }

        void reset() override
        {
        }
    };
}