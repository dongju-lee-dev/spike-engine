#pragma once

#include <any>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <map>
#include <utility>

#include "snn/core/memory.hpp"

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

    class LearningLUT
    {
    public:
        static const F32Buffer *get_or_create(float tau, float max_ratio)
        {
            static std::mutex mutex;
            static std::map<std::pair<float, float>, F32Buffer> cache;

            std::lock_guard<std::mutex> lock(mutex);
            const auto key = std::make_pair(tau, max_ratio);

            auto it = cache.find(key);
            if (it != cache.end())
            {
                return &it->second;
            }

            F32Buffer buffer;
            if (tau <= 0.0f)
            {
                buffer.assign(1, 1.0f);
            }
            else
            {
                const std::size_t count = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(tau * max_ratio)));
                buffer.resize(count);
                for (std::size_t dt = 0; dt < count; ++dt)
                {
                    buffer[dt] = std::exp(-static_cast<float>(dt) / tau);
                }
            }

            auto [inserted_it, _] = cache.emplace(key, std::move(buffer));
            return &inserted_it->second;
        }
    };
}