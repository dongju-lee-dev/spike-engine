#pragma once

#include <any>

namespace snn
{
    class Loader
    {
    public:
        virtual ~Loader() = default;

        virtual void reset() = 0;
        virtual bool next() = 0;
        virtual std::any get() = 0;
    };
}