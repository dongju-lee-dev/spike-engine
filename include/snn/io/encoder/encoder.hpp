#pragma once

#include <any>

#include "snn/core/memory.hpp"

namespace snn
{
    class Encoder
    {
    public:
        virtual ~Encoder() = default;

        virtual std::vector<F32Buffer> encode(const std::any &raw_input) = 0;
    };
}