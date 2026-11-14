#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "snn/core/memory.hpp"
#include "snn/neuron/neuron.hpp"
#include "snn/synapse/synapse.hpp"

namespace snn
{
    class Network
    {
    public:
        std::vector<std::unique_ptr<NeuronContext>> neu_context_;
        std::vector<std::unique_ptr<SynapseContext>> syn_context_;

        explicit Network() = default;
        ~Network() = default;

        std::size_t add_neuron(std::unique_ptr<NeuronContext> neuron_context)
        {
            neu_context_.push_back(std::move(neuron_context));
            return neu_context_.size() - 1;
        }

        std::size_t add_synapse(std::unique_ptr<SynapseContext> synapse_context)
        {
            syn_context_.push_back(std::move(synapse_context));
            return syn_context_.size() - 1;
        }

        void init()
        {
            neu_context_.shrink_to_fit();
            syn_context_.shrink_to_fit();

            for (const auto &s : syn_context_)
            {
                if (s->pre_id >= neu_context_.size() || s->post_id >= neu_context_.size())
                {
                    std::cerr << "ERROR: Invalid neuron ID in synapse context.\n";
                    std::exit(EXIT_FAILURE);
                }

                const std::size_t pre_size = neu_context_[s->pre_id]->neuron->size();
                const std::size_t post_size = neu_context_[s->post_id]->neuron->size();

                if (s->synapse->pre_size() != pre_size || s->synapse->post_size() != post_size)
                {
                    std::cerr << "ERROR: Synapse and Neuron size mismatch.\n"
                              << "Expected pre: " << pre_size << ", got: " << s->synapse->pre_size() << "\n"
                              << "Expected post: " << post_size << ", got: " << s->synapse->post_size() << "\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        void reset()
        {
            for (const auto &n : neu_context_)
            {
                n->neuron->reset();
                n->current.clear();
                n->spike.clear();
            }
        }

        void step(const F32Buffer &input, BitBuffer &output)
        {
            if (!input.empty())
            {
                float *SNN_RESTRICT cur_ptr = SNN_ASSUME_ALIGNED(neu_context_[0]->current.data(), 64);
                const float *SNN_RESTRICT inp_ptr = SNN_ASSUME_ALIGNED(input.data(), 64);

                const std::size_t len = input.size();

#pragma omp simd aligned(cur_ptr, inp_ptr : 64)
                for (std::size_t i = 0; i < len; ++i)
                {
                    cur_ptr[i] += inp_ptr[i];
                }
            }

            for (auto &s : syn_context_)
            {
                s->synapse->step(neu_context_[s->pre_id]->spike, neu_context_[s->post_id]->current);
            }

            for (auto &n : neu_context_)
            {
                n->neuron->step(n->current, n->spike);
                std::fill(n->current.begin(), n->current.end(), 0.0f);
            }

            output.copy_from(neu_context_.back()->spike);
        }

        [[nodiscard]] bool is_active() const noexcept
        {
            for (const auto &n : neu_context_)
            {
                for (const auto &d : n->spike.data_)
                {
                    if (d != 0ULL)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        [[nodiscard]] NeuronContext &get_neuron_context(std::size_t id)
        {
            return *neu_context_[id];
        }

        [[nodiscard]] SynapseContext &get_synapse_context(std::size_t id)
        {
            return *syn_context_[id];
        }
    };
}