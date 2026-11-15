#pragma once

#include <algorithm>

#include "snn/learning/learning.hpp"
#include "snn/neuron/neuron.hpp"
#include "snn/synapse/synapse.hpp"
#include "snn/synapse/synapse_utils.hpp"

namespace snn
{
    enum class STDPKernel
    {
        Pair,
        Triplet,
    };

    enum class STDPWeight
    {
        Add,
        Multiply,
    };

    enum class STDPMode
    {
        Asymmetric,
        Symmetric,
    };

    template <STDPKernel kernel, STDPWeight weight, STDPMode mode>
    class STDP : public LearningRule
    {
    public:
        NeuronContext *nc_0_;
        NeuronContext *nc_1_;
        SynapseContext *sc_0_;

        U32Buffer ts_0_;
        U32Buffer ts_1_;

        float w_max_;
        float w_min_;
        float a_plus_;
        float a_minus_;
        float at_plus_;
        float at_minus_;

        const F32Buffer *ltp_lut_;
        const F32Buffer *ltd_lut_;
        const F32Buffer *ltpt_lut_;
        const F32Buffer *ltdt_lut_;

        template <typename = std::enable_if_t<kernel == STDPKernel::Pair>>
        explicit STDP(NeuronContext &nc_0, NeuronContext &nc_1, SynapseContext &sc_0,
                      float w_max, float w_min, float a_plus, float a_minus, float t_plus, float t_minus, float max_ratio)
            : nc_0_(&nc_0),
              nc_1_(&nc_1),
              sc_0_(&sc_0),
              ts_0_(nc_0.neuron->size(), 0),
              ts_1_(nc_1.neuron->size(), 0),
              w_max_(w_max),
              w_min_(w_min),
              a_plus_(a_plus),
              a_minus_(a_minus),
              at_plus_(0.0f),
              at_minus_(0.0f),
              ltp_lut_(LearningLUT::get_or_create(t_plus, max_ratio)),
              ltd_lut_(LearningLUT::get_or_create(t_minus, max_ratio)),
              ltpt_lut_(nullptr),
              ltdt_lut_(nullptr)
        {
        }

        template <typename = std::enable_if_t<kernel == STDPKernel::Triplet>>
        explicit STDP(NeuronContext &nc_0, NeuronContext &nc_1, SynapseContext &sc_0,
                      float w_max, float w_min, float a_plus, float a_minus, float t_plus, float t_minus, float max_ratio,
                      float at_plus, float at_minus, float tt_plus, float tt_minus, float maxt_ratio)
            : nc_0_(&nc_0),
              nc_1_(&nc_1),
              sc_0_(&sc_0),
              ts_0_(nc_0.neuron->size(), 0),
              ts_1_(nc_1.neuron->size(), 0),
              w_max_(w_max),
              w_min_(w_min),
              a_plus_(a_plus),
              a_minus_(a_minus),
              at_plus_(at_plus),
              at_minus_(at_minus),
              ltp_lut_(LearningLUT::get_or_create(t_plus, max_ratio)),
              ltd_lut_(LearningLUT::get_or_create(t_minus, max_ratio)),
              ltpt_lut_(LearningLUT::get_or_create(tt_plus, maxt_ratio)),
              ltdt_lut_(LearningLUT::get_or_create(tt_minus, maxt_ratio))
        {
        }

        void start(const std::any &) override
        {
        }

        void step(const std::size_t tick) override
        {
            const std::uint32_t cur_tick = static_cast<std::uint32_t>(tick);

            if constexpr (kernel == STDPKernel::Pair)
            {
                step_pair(cur_tick);
            }
            else if constexpr (kernel == STDPKernel::Triplet)
            {
                step_triplet(cur_tick);
            }
        }

        void end(const std::any &) override
        {
        }

        void reset() override
        {
            std::fill(ts_0_.begin(), ts_0_.end(), 0);
            std::fill(ts_1_.begin(), ts_1_.end(), 0);
        }

    private:
        inline void update_weight(float &w, const float delta) const noexcept
        {
            if constexpr (weight == STDPWeight::Add)
            {
                w = std::clamp(w + delta, w_min_, w_max_);
            }
            else if constexpr (weight == STDPWeight::Multiply)
            {
                if (delta > 0.0f)
                {
                    w += delta * (w_max_ - w);
                }
                else
                {
                    w += delta * (w - w_min_);
                }
                w = std::clamp(w, w_min_, w_max_);
            }
        }

        inline void step_pair(const std::uint32_t tick)
        {
            Synapse &syn = *sc_0_->synapse;
            float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(syn.weights().data(), 64);

            const std::uint32_t *SNN_RESTRICT ts_0_ptr = SNN_ASSUME_ALIGNED(ts_0_.data(), 64);
            const std::uint32_t *SNN_RESTRICT ts_1_ptr = SNN_ASSUME_ALIGNED(ts_1_.data(), 64);

            const float *SNN_RESTRICT ltp_ptr = SNN_ASSUME_ALIGNED(ltp_lut_->data(), 64);
            const std::size_t ltp_sz = ltp_lut_->size();
            const float *SNN_RESTRICT ltd_ptr = SNN_ASSUME_ALIGNED(ltd_lut_->data(), 64);
            const std::size_t ltd_sz = ltd_lut_->size();

            nc_1_->spike.for_each_true([&](const std::size_t post_id)
                                       { syn.for_each_pre(post_id, [&](const std::size_t pre_id, const std::size_t w_id)
                                                          {
                        const std::uint32_t t_pre = ts_0_ptr[pre_id];
                        if (t_pre != 0)
                        {
                            const std::uint32_t dt = tick - t_pre;
                            if (dt < ltp_sz)
                            {
                                const float delta = a_plus_ * ltp_ptr[dt];
                                update_weight(w_ptr[w_id], delta);
                            }
                        } }); });

            nc_0_->spike.for_each_true([&](const std::size_t pre_id)
                                       { syn.for_each_post(pre_id, [&](const std::size_t post_id, const std::size_t w_id)
                                                           {
                        const std::uint32_t t_post = ts_1_ptr[post_id];
                        if (t_post != 0)
                        {
                            const std::uint32_t dt = tick - t_post;
                            if constexpr (mode == STDPMode::Asymmetric)
                            {
                                if (dt < ltd_sz)
                                {
                                    const float delta = -a_minus_ * ltd_ptr[dt];
                                    update_weight(w_ptr[w_id], delta);
                                }
                            }
                            else if constexpr (mode == STDPMode::Symmetric)
                            {
                                if (dt < ltp_sz)
                                {
                                    const float delta = a_plus_ * ltp_ptr[dt];
                                    update_weight(w_ptr[w_id], delta);
                                }
                            }
                        } }); });

            nc_1_->spike.for_each_true([&](const std::size_t post_id)
                                       { ts_1_[post_id] = tick; });

            nc_0_->spike.for_each_true([&](const std::size_t pre_id)
                                       { ts_0_[pre_id] = tick; });
        }

        inline void step_triplet(const std::uint32_t tick)
        {
            Synapse &syn = *sc_0_->synapse;
            float *SNN_RESTRICT w_ptr = SNN_ASSUME_ALIGNED(syn.weights().data(), 64);

            const std::uint32_t *SNN_RESTRICT ts_0_ptr = SNN_ASSUME_ALIGNED(ts_0_.data(), 64);
            const std::uint32_t *SNN_RESTRICT ts_1_ptr = SNN_ASSUME_ALIGNED(ts_1_.data(), 64);

            const float *SNN_RESTRICT ltp_ptr = SNN_ASSUME_ALIGNED(ltp_lut_->data(), 64);
            const std::size_t ltp_sz = ltp_lut_->size();

            const float *SNN_RESTRICT ltd_ptr = SNN_ASSUME_ALIGNED(ltd_lut_->data(), 64);
            const std::size_t ltd_sz = ltd_lut_->size();

            const float *SNN_RESTRICT ltpt_ptr = SNN_ASSUME_ALIGNED(ltpt_lut_->data(), 64);
            const std::size_t ltpt_sz = ltpt_lut_->size();

            const float *SNN_RESTRICT ltdt_ptr = SNN_ASSUME_ALIGNED(ltdt_lut_->data(), 64);
            const std::size_t ltdt_sz = ltdt_lut_->size();

            nc_1_->spike.for_each_true([&](const std::size_t post_id)
                                       {
                    float post_factor = a_plus_;
                    const std::uint32_t t_post_prev = ts_1_ptr[post_id];
                    if (t_post_prev != 0)
                    {
                        const std::uint32_t dt_post = tick - t_post_prev;
                        if (dt_post < ltpt_sz)
                        {
                            post_factor += at_plus_ * ltpt_ptr[dt_post];
                        }
                    }
    
                    syn.for_each_pre(post_id, [&](const std::size_t pre_id, const std::size_t w_id)
                    {
                        const std::uint32_t t_pre = ts_0_ptr[pre_id];
                        if (t_pre != 0)
                        {
                            const std::uint32_t dt_pre = tick - t_pre;
                            if (dt_pre < ltp_sz)
                            {
                                const float delta = post_factor * ltp_ptr[dt_pre];
                                update_weight(w_ptr[w_id], delta);
                            }
                        }
                    }); });

            nc_0_->spike.for_each_true([&](const std::size_t pre_id)
                                       {
                    float pre_factor = a_minus_;
                    const std::uint32_t t_pre_prev = ts_0_ptr[pre_id];
                    if (t_pre_prev != 0)
                    {
                        const std::uint32_t dt_pre = tick - t_pre_prev;
                        if (dt_pre < ltdt_sz)
                        {
                            pre_factor += at_minus_ * ltdt_ptr[dt_pre];
                        }
                    }

                    syn.for_each_post(pre_id, [&](const std::size_t post_id, const std::size_t w_id)
                    {
                        const std::uint32_t t_post = ts_1_ptr[post_id];
                        if (t_post != 0)
                        {
                            const std::uint32_t dt_post = tick - t_post;
                            if constexpr (mode == STDPMode::Asymmetric)
                            {
                                if (dt_post < ltd_sz)
                                {
                                    const float delta = -pre_factor * ltd_ptr[dt_post];
                                    update_weight(w_ptr[w_id], delta);
                                }
                            }
                            else if constexpr (mode == STDPMode::Symmetric)
                            {
                                if (dt_post < ltp_sz)
                                {
                                    const float delta = pre_factor * ltp_ptr[dt_post];
                                    update_weight(w_ptr[w_id], delta);
                                }
                            }
                        }
                    }); });

            nc_1_->spike.for_each_true([&](const std::size_t post_id)
                                       { ts_1_[post_id] = tick; });

            nc_0_->spike.for_each_true([&](const std::size_t pre_id)
                                       { ts_0_[pre_id] = tick; });
        }
    };
}