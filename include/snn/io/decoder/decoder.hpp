#pragma once

#include <any>

#include "snn/core/memory.hpp"

namespace snn
{
    class Decoder
    {
    public:
        virtual ~Decoder() = default;

        virtual std::any decode(const std::vector<BitBuffer> &row_output) = 0;
    };
}