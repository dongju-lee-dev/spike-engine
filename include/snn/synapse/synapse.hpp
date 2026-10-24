#pragma once

#include <cstddef>
#include "snn/core/memory.hpp"

namespace snn
{
    enum class SynapseMode
    {
        Front,
        Back
    };

    class Synapse
    {
    public:
        virtual ~Synapse() = default;

        virtual void step(const F32Buffer &pre_spikes, F32Buffer &post_current) = 0;

        [[nodiscard]] virtual SynapseMode mode() const noexcept = 0;
        [[nodiscard]] virtual std::size_t pre_size() const noexcept = 0;
        [[nodiscard]] virtual std::size_t post_size() const noexcept = 0;

        [[nodiscard]] virtual F32Buffer &weights() noexcept = 0;
        [[nodiscard]] virtual const F32Buffer &weights() const noexcept = 0;
    };
}