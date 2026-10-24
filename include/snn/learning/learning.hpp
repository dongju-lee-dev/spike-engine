#pragma once

#include <any>
#include <cstddef>

namespace snn
{
    class LearningRule
    {
    public:
        virtual ~LearningRule() = default;

        virtual void start(const std::any &input) = 0;
        virtual void step(const std::size_t tick) = 0;
        virtual void end(const std::any &output) = 0;
        virtual void reset() = 0;
    };
}