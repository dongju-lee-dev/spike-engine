#pragma once

#include <cstddef>
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
}