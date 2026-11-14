#pragma once

#include <cstddef>
#include <memory>
#include "snn/core/memory.hpp"

namespace snn
{
    class Neuron
    {
    public:
        virtual ~Neuron() = default;

        virtual void step(const F32Buffer &current, BitBuffer &spike) = 0;
        virtual void reset() = 0;

        [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    };

    class NeuronContext
    {
    public:
        std::unique_ptr<Neuron> neuron;
        F32Buffer current;
        BitBuffer spike;

        explicit NeuronContext(std::unique_ptr<Neuron> neu)
            : neuron(std::move(neu)),
              current(neuron->size(), 0.0),
              spike(neuron->size(), false)
        {
        }
    };
}