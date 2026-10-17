#include "snn/core/memory.hpp"

namespace snn
{
    class Synapse
    {
    public:
        virtual ~Synapse() = default;

        virtual void step(const F32Buffer &spkie_buffer, F32Buffer &buffer) = 0;
    };
}