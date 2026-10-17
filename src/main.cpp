#include <iostream>
#include <iomanip>
#include "snn/neuron/neuron_if.hpp"
#include "snn/neuron/neuron_lif.hpp"

void test_if_neuron()
{
    std::cout << "\n========================================\n";
    std::cout << " [1] IF Neuron Test (No Leak)\n";
    std::cout << " - Threshold: 1.0, Reset: 0.0\n";
    std::cout << "========================================\n";

    constexpr std::size_t N = 2; // 뉴런 2개
    snn::IFConfig config{.v_threshold = 1.0f, .v_reset = 0.0f};
    snn::IFNeuron neuron(N, config);

    // 입출력 겸용 In-place 버퍼 (뉴런 0번에는 0.4, 뉴런 1번에는 0.0 주입)
    snn::F32Buffer buffer(N);

    for (int tick = 1; tick <= 5; ++tick)
    {
        // 전류 주입 설정 (1번 뉴런만 0.4씩 계속 주입)
        buffer[0] = 0.4f;
        buffer[1] = 0.0f;

        // 1틱 실행 (In-place)
        neuron.step(buffer);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[Tick " << tick << "] "
                  << "N0: (Spike=" << buffer[0] << ", V=" << neuron.v_[0] << ") | "
                  << "N1: (Spike=" << buffer[1] << ", V=" << neuron.v_[1] << ")\n";
    }
}

void test_lif_neuron()
{
    std::cout << "\n========================================\n";
    std::cout << " [2] LIF Neuron Test (Leaky, Decay = 0.5)\n";
    std::cout << " - Threshold: 1.0, Reset: 0.0, Decay: 0.5\n";
    std::cout << "========================================\n";

    constexpr std::size_t N = 2;
    snn::LIFConfig config{.v_threshold = 1.0f, .v_reset = 0.0f, .decay = 0.5f};
    snn::LIFNeuron neuron(N, config);

    snn::F32Buffer buffer(N);

    // 시나리오:
    // Tick 1: 강한 자극(0.8) 주입 -> 전위 0.8 도달
    // Tick 2~3: 입력 전류 0 주입 -> 누수(Leak)로 전위가 절반씩 감쇠하는지 확인
    // Tick 4: 1.0 이상의 전류 주입 -> 즉시 발화 및 리셋 확인
    float inputs[5] = {0.8f, 0.0f, 0.0f, 1.2f, 0.0f};

    for (int tick = 1; tick <= 5; ++tick)
    {
        buffer[0] = inputs[tick - 1];
        buffer[1] = 0.0f;

        neuron.step(buffer);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[Tick " << tick << "] Input=" << inputs[tick - 1] << " -> "
                  << "N0: (Spike=" << buffer[0] << ", V=" << neuron.v_[0] << ")\n";
    }
}

int main()
{
    std::cout << ">>> SNN Neuron Simulation Verification <<<\n";

    test_if_neuron();
    test_lif_neuron();

    std::cout << "\n>>> All tests completed successfully! <<<\n";
    return 0;
}