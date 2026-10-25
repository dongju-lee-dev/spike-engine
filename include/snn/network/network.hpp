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
        struct NeuronBox
        {
            std::unique_ptr<Neuron> neu;
            F32Buffer current;
            BitBuffer spike;
        };

        struct SynapseBox
        {
            std::unique_ptr<Synapse> syn;
            std::size_t pre_id;
            std::size_t post_id;
        };

        std::vector<NeuronBox> neu_;
        std::vector<SynapseBox> syn_;

        Network() = default;

        std::size_t add_neuron(std::unique_ptr<Neuron> neuron)
        {
            const std::size_t sz = neuron->size();
            neu_.push_back(NeuronBox{
                .neu = std::move(neuron),
                .current = F32Buffer(sz, 0.0f),
                .spike = BitBuffer(sz, false),
            });
            return neu_.size() - 1;
        }

        std::size_t add_synapse(std::unique_ptr<Synapse> &synapse, std::size_t pre_id, std::size_t post_id)
        {
            syn_.push_back(SynapseBox{
                .syn = std::move(synapse),
                .pre_id = pre_id,
                .post_id = post_id,
            });
            return syn_.size() - 1;
        }

        void init()
        {
            neu_.shrink_to_fit();
            syn_.shrink_to_fit();

            for (const auto &s : syn_)
            {
                if (s.syn->pre_size() != neu_[s.pre_id].neu->size() || s.syn->post_size() != neu_[s.post_id].neu->size())
                {
                    std::cerr << "ERROR : The sizes do not match.\n (pre neuron size : " << neu_[s.pre_id].neu->size() << ", post neuron size : " << neu_[s.post_id].neu->size() << ")\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        void reset()
        {
            for (const auto &n : neu_)
            {
                n.neu->reset();
            }
        }

        void step(const F32Buffer &input, BitBuffer &output)
        {
            if (!input.empty())
            {
                float *SNN_RESTRICT cur_ptr = neu_[0].current.data();
                const float *SNN_RESTRICT in_ptr = input.data();
                const std::size_t len = std::min(neu_[0].current.size(), input.size());

#pragma omp simd aligned(cur_ptr, in_ptr : 64)
                for (std::size_t i = 0; i < len; ++i)
                {
                    cur_ptr[i] += in_ptr[i];
                }
            }

            for (auto &s : syn_)
            {
                s.syn->step(neu_[s.pre_id].spike, neu_[s.post_id].current);
            }

            for (auto &n : neu_)
            {
                n.neu->step(n.current, n.spike);
                std::fill(n.current.begin(), n.current.end(), 0.0f);
            }

            if (!output.empty())
            {
                output.copy_from(neu_.back().spike);
            }
        }

        [[nodiscard]] NeuronBox &neuron_box(std::size_t id)
        {
            if (id >= neu_.size())
                throw std::out_of_range("Invalid neuron box ID");
            return neu_[id];
        }

        [[nodiscard]] const NeuronBox &neuron_box(std::size_t id) const
        {
            if (id >= neu_.size())
                throw std::out_of_range("Invalid neuron box ID");
            return neu_[id];
        }

        [[nodiscard]] SynapseBox &synapse_box(std::size_t id)
        {
            if (id >= syn_.size())
                throw std::out_of_range("Invalid synapse box ID");
            return syn_[id];
        }

        [[nodiscard]] const SynapseBox &synapse_box(std::size_t id) const
        {
            if (id >= syn_.size())
                throw std::out_of_range("Invalid synapse box ID");
            return syn_[id];
        }
    };
}