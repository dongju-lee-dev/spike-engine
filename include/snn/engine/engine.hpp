#pragma once

#include <any>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <iomanip>
#include <iostream>

#include "snn/network/network.hpp"
#include "snn/io/loader/loader.hpp"
#include "snn/io/encoder/encoder.hpp"
#include "snn/io/decoder/decoder.hpp"
#include "snn/learning/learning.hpp"

namespace snn
{
    class Engine
    {
    public:
        std::unique_ptr<Loader> loader_;
        std::unique_ptr<Encoder> encoder_;
        std::unique_ptr<Decoder> decoder_;
        std::unique_ptr<Network> network_;

        std::vector<std::unique_ptr<LearningRule>> learning_rules_;

        Engine(
            std::unique_ptr<Loader> loader,
            std::unique_ptr<Encoder> encoder,
            std::unique_ptr<Decoder> decoder,
            std::unique_ptr<Network> network,
            std::vector<std::unique_ptr<LearningRule>> learning_rules)
            : loader_(std::move(loader)),
              encoder_(std::move(encoder)),
              decoder_(std::move(decoder)),
              network_(std::move(network)),
              learning_rules_(std::move(learning_rules))
        {
            network_->init();

            unsigned int n = std::thread::hardware_concurrency();

            std::cout << "cores : " << n << '\n'
                      << std::endl;
        }

        ~Engine() = default;

    private:
        bool run_network()
        {
            for (const auto &n : network_->neu_)
            {
                for (const auto &d : n.spike.data_)
                {
                    if (d != 0ULL)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

    public:
        void run(std::size_t epoch)
        {
            auto start_time = std::chrono::steady_clock::now();

            for (std::size_t e = 0; e < epoch; ++e)
            {
                loader_->reset();

                std::size_t last_p = 0;

                for (std::size_t count = 0; loader_->next(); count++)
                {
                    network_->reset();

                    for (const auto &lr : learning_rules_)
                    {
                        lr->reset();
                    }

                    const std::any row_data = loader_->get();

                    for (const auto &lr : learning_rules_)
                    {
                        lr->start(row_data);
                    }

                    const std::vector<F32Buffer> input_data = encoder_->encode(row_data);
                    const std::size_t input_len = input_data.size();
                    std::vector<BitBuffer> output_data;

                    for (std::size_t tick = 0; run_network(); ++tick)
                    {
                        output_data.push_back(BitBuffer(input_len, false));

                        network_->step(tick < input_len ? input_data[tick] : F32Buffer(), output_data.back());

                        for (const auto &lr : learning_rules_)
                        {
                            lr->step(tick);
                        }
                    }

                    const std::any out_row_data = decoder_->decode(output_data);

                    for (const auto &lr : learning_rules_)
                    {
                        lr->end(out_row_data);
                    }

                    std::size_t current_p = (count * 100) / loader_->len();

                    if (current_p > last_p)
                    {
                        last_p = current_p;

                        auto now = std::chrono::steady_clock::now();
                        std::chrono::duration<double> elapsed = now - start_time;
                        long long total_sec = static_cast<long long>(elapsed.count());

                        long long h = total_sec / 3600;
                        long long m = (total_sec % 3600) / 60;
                        long long s = total_sec % 60;

                        std::cout << "\r"
                                  << e << "/" << epoch << ", "
                                  << count << "/" << loader_->len() << ", "
                                  << std::setw(3) << current_p << "%, "
                                  << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s
                                  << "          "
                                  << std::flush;
                    }
                }

                std::cout << "\n"
                          << std::flush;
            }
        }
    };
}