#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "snn/synapse/synapse.hpp"
#include "snn/synapse/synapse_dense.hpp"
#include "snn/synapse/synapse_sparse.hpp"

namespace snn
{
    inline void init_uniform(Synapse &syn, float min_w, float max_w, std::mt19937::result_type seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_w, max_w);
        for (float &w : syn.weights())
        {
            w = dist(gen);
        }
    }

    inline void init_normal(Synapse &syn, float mean, float stddev, std::mt19937::result_type seed = 42)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(mean, stddev);
        for (float &w : syn.weights())
        {
            w = dist(gen);
        }
    }

    inline void init_constant(Synapse &syn, float value)
    {
        auto &w = syn.weights();
        std::fill(w.begin(), w.end(), value);
    }

    inline void connect_uniform_ratio(SparseSynapse &syn, float ratio)
    {
        const std::size_t nf = (syn.mode() == SynapseMode::Front) ? syn.pre_size() : syn.post_size();
        const std::size_t nb = (syn.mode() == SynapseMode::Front) ? syn.post_size() : syn.pre_size();
        const std::size_t k = std::clamp(static_cast<std::size_t>(std::round(nf * ratio)), std::size_t{1}, nf);

        syn.c_.clear();
        syn.i_.clear();
        syn.w_.clear();

        syn.c_.resize(nf);
        syn.i_.reserve(nf * k);
        syn.w_.reserve(nf * k);

        const double step = static_cast<double>(nf) / static_cast<double>(k);
        const double phi = step / static_cast<double>(nb);

        for (std::size_t j = 0; j < nf; ++j)
        {
            syn.c_[j] = k;

            for (std::size_t idx = 0; idx < k; ++idx)
            {
                std::size_t target_idx = static_cast<std::size_t>(idx * step + j * phi) % nb;

                syn.i_.push_back(target_idx);
                syn.w_.push_back(0.0f);
            }
        }
    }

    inline void connect_random_ratio(SparseSynapse &syn, float ratio, std::mt19937::result_type seed = 42)
    {
        const std::size_t nf = (syn.mode() == SynapseMode::Front) ? syn.pre_size() : syn.post_size();
        const std::size_t nb = (syn.mode() == SynapseMode::Front) ? syn.post_size() : syn.pre_size();
        const std::size_t k = std::clamp(static_cast<std::size_t>(std::round(nb * ratio)), std::size_t{1}, nb);

        syn.c_.clear();
        syn.i_.clear();
        syn.w_.clear();

        syn.c_.resize(nf);
        syn.i_.reserve(nf * k);
        syn.w_.reserve(nf * k);

        std::mt19937 gen(seed);
        std::vector<std::size_t> indices(nb);
        for (std::size_t i = 0; i < nb; ++i)
        {
            indices[i] = i;
        }

        for (std::size_t j = 0; j < nf; ++j)
        {
            syn.c_[j] = k;

            std::shuffle(indices.begin(), indices.end(), gen);

            for (std::size_t idx = 0; idx < k; ++idx)
            {
                syn.i_.push_back(indices[idx]);
                syn.w_.push_back(0.0f);
            }
        }

        syn.c_.shrink_to_fit();
        syn.i_.shrink_to_fit();
        syn.w_.shrink_to_fit();
    }

    inline void connect_random(SparseSynapse &syn, std::mt19937::result_type seed = 42)
    {
        const std::size_t nf = (syn.mode() == SynapseMode::Front) ? syn.pre_size() : syn.post_size();
        const std::size_t nb = (syn.mode() == SynapseMode::Front) ? syn.post_size() : syn.pre_size();

        syn.c_.clear();
        syn.i_.clear();
        syn.w_.clear();

        syn.c_.resize(nf);

        std::mt19937 gen(seed);
        std::uniform_int_distribution<std::size_t> count_dist(1, nb);
        std::vector<std::size_t> indices(nb);
        for (std::size_t i = 0; i < nb; ++i)
        {
            indices[i] = i;
        }

        syn.i_.reserve(nf * (nb / 2));
        syn.w_.reserve(nf * (nb / 2));

        for (std::size_t j = 0; j < nf; ++j)
        {
            std::size_t k = count_dist(gen);
            syn.c_[j] = k;

            std::shuffle(indices.begin(), indices.end(), gen);

            for (std::size_t idx = 0; idx < k; ++idx)
            {
                syn.i_.push_back(indices[idx]);
                syn.w_.push_back(0.0f);
            }
        }

        syn.c_.shrink_to_fit();
        syn.i_.shrink_to_fit();
        syn.w_.shrink_to_fit();
    }

    inline void connect_gaussian(SparseSynapse &syn, float sigma, float ratio = 0.3f, std::mt19937::result_type seed = 42)
    {
        const std::size_t n_src = (syn.mode() == SynapseMode::Front) ? syn.pre_size() : syn.post_size();
        const std::size_t n_pool = (syn.mode() == SynapseMode::Front) ? syn.post_size() : syn.pre_size();
        const std::size_t k = std::clamp(static_cast<std::size_t>(std::round(n_pool * ratio)), std::size_t{1}, n_pool);

        syn.c_.clear();
        syn.i_.clear();
        syn.w_.clear();

        syn.c_.resize(n_src);
        syn.i_.reserve(n_src * k);
        syn.w_.reserve(n_src * k);

        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, sigma);
        std::vector<bool> picked(n_pool, false);

        std::vector<std::size_t> current_indices(k);

        for (std::size_t i = 0; i < n_src; ++i)
        {
            syn.c_[i] = k;

            const float center = (static_cast<float>(i) + 0.5f) * (static_cast<float>(n_pool) / static_cast<float>(n_src));

            std::size_t count = 0;
            std::size_t attempts = 0;
            const std::size_t max_attempts = k * 20;

            while (count < k && attempts < max_attempts)
            {
                ++attempts;
                float sample = center + dist(gen);
                long long idx = static_cast<long long>(std::round(sample)) % static_cast<long long>(n_pool);
                if (idx < 0)
                    idx += n_pool;

                std::size_t uidx = static_cast<std::size_t>(idx);
                if (!picked[uidx])
                {
                    picked[uidx] = true;
                    current_indices[count] = uidx;
                    ++count;
                }
            }

            if (count < k)
            {
                for (std::size_t step = 0; step < n_pool && count < k; ++step)
                {
                    long long left = (static_cast<long long>(std::round(center)) - static_cast<long long>(step)) % static_cast<long long>(n_pool);
                    if (left < 0)
                        left += n_pool;

                    std::size_t uleft = static_cast<std::size_t>(left);
                    if (!picked[uleft])
                    {
                        picked[uleft] = true;
                        current_indices[count++] = uleft;
                        if (count == k)
                            break;
                    }

                    long long right = (static_cast<long long>(std::round(center)) + static_cast<long long>(step)) % static_cast<long long>(n_pool);
                    if (right < 0)
                        right += n_pool;

                    std::size_t uright = static_cast<std::size_t>(right);
                    if (!picked[uright])
                    {
                        picked[uright] = true;
                        current_indices[count++] = uright;
                        if (count == k)
                            break;
                    }
                }
            }

            for (std::size_t j = 0; j < k; ++j)
            {
                syn.i_.push_back(current_indices[j]);
                syn.w_.push_back(0.0f);
            }

            for (std::size_t j = 0; j < k; ++j)
            {
                picked[current_indices[j]] = false;
            }
        }

        syn.c_.shrink_to_fit();
        syn.i_.shrink_to_fit();
        syn.w_.shrink_to_fit();
    }
}