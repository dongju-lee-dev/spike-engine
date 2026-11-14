#include <memory>

#include "snn/neuron/neuron_if.hpp"
#include "snn/synapse/synapse_dense.hpp"
#include "snn/synapse/synapse_utils.hpp"
#include "snn/network/network.hpp"
#include "snn/learning/stdp.hpp"
#include "snn/engine/engine.hpp"

int main()
{
    auto n_0 = std::make_unique<snn::IFNeuron>(2, snn::IFConfig{.v_threshold = 1.0f, .v_reset = 0.0f});
    auto n_1 = std::make_unique<snn::IFNeuron>(10, snn::IFConfig{.v_threshold = 1.0f, .v_reset = 0.0f});
    auto n_2 = std::make_unique<snn::IFNeuron>(1, snn::IFConfig{.v_threshold = 1.0f, .v_reset = 0.0f});

    auto s_0 = std::make_unique<snn::DenseSynapse>(2, 10);
    auto s_1 = std::make_unique<snn::DenseSynapse>(10, 1);

    snn::init_uniform(*s_0, -0.2f, 0.2f);
    snn::init_uniform(*s_1, -0.2f, 0.2f);

    auto net = std::make_unique<snn::Network>();

    auto n_0_id = net->add_neuron(std::make_unique<snn::NeuronContext>(std::move(n_0)));
    auto n_1_id = net->add_neuron(std::make_unique<snn::NeuronContext>(std::move(n_1)));
    auto n_2_id = net->add_neuron(std::make_unique<snn::NeuronContext>(std::move(n_2)));

    net->add_synapse(std::make_unique<snn::SynapseContext>(std::move(s_0), n_0_id, n_1_id));
    net->add_synapse(std::make_unique<snn::SynapseContext>(std::move(s_1), n_1_id, n_2_id));

    net->init();

    // 로더 생성
    // 인코더 생성
    // 디코더 생성
    // 학습 규칙 생성
    
    auto engine = snn::Engine(NULL, NULL, NULL, std::move(net), {});

    // 실행

    return 0;
}