#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "snn/core/memory.hpp"

namespace snn
{
    struct SynapseCallback
    {
        void *ctx;
        void (*invoker)(void *, std::size_t, std::size_t);

        template <typename F>
        SynapseCallback(F &&f) noexcept
            : ctx(const_cast<void *>(reinterpret_cast<const void *>(std::addressof(f)))),
              invoker([](void *c, std::size_t neuron_id, std::size_t weight_id)
                      { (*reinterpret_cast<std::remove_reference_t<F> *>(c))(neuron_id, weight_id); })
        {
        }

        inline void operator()(std::size_t neuron_id, std::size_t weight_id) const
        {
            invoker(ctx, neuron_id, weight_id);
        }
    };

    class Synapse
    {
    public:
        virtual ~Synapse() = default;

        virtual void step(const BitBuffer &spike, F32Buffer &current) = 0;

        virtual void for_each_post_impl(std::size_t pre_id, SynapseCallback cb) = 0;
        virtual void for_each_pre_impl(std::size_t post_id, SynapseCallback cb) = 0;

        template <typename F>
        inline void for_each_post(std::size_t pre_id, F &&func)
        {
            for_each_post_impl(pre_id, SynapseCallback(func));
        }

        template <typename F>
        inline void for_each_pre(std::size_t post_id, F &&func)
        {
            for_each_pre_impl(post_id, SynapseCallback(func));
        }

        [[nodiscard]] virtual std::size_t pre_size() const noexcept = 0;
        [[nodiscard]] virtual std::size_t post_size() const noexcept = 0;

        [[nodiscard]] virtual F32Buffer &weights() noexcept = 0;
        [[nodiscard]] virtual const F32Buffer &weights() const noexcept = 0;
    };

    class SynapseContext
    {
    public:
        std::unique_ptr<Synapse> synapse;
        std::size_t pre_id;
        std::size_t post_id;

        explicit SynapseContext(std::unique_ptr<Synapse> syn, std::size_t pri, std::size_t poi)
            : synapse(std::move(syn)),
              pre_id(pri),
              post_id(poi)
        {
        }
    };
}