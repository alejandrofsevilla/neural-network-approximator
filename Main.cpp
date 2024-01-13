#include <opennn/data_set.h>
#include <opennn/neural_network.h>
#include <opennn/perceptron_layer.h>
#include <opennn/scaling_layer.h>
#include <opennn/training_strategy.h>

#include <boost/program_options.hpp>
#include <cstdlib>
#include <fstream>

#include "opennn/bounding_layer.h"

namespace {

namespace po = boost::program_options;

inline auto makeDataSet(auto function, auto minInput, auto maxInput,
                        auto numberOfSamples) {
  auto ds{std::make_shared<opennn::DataSet>(numberOfSamples, 2)};
  for (auto i = 0; i < numberOfSamples; i++) {
    auto x{minInput + i * (maxInput - minInput) / numberOfSamples};
    ds->get_data_pointer()->operator()(i, 0) = x;
    ds->get_data_pointer()->operator()(i, 1) = function(x);
  }
  return ds;
}

inline auto generateDataSetFiles(auto path) {
  auto cos{makeDataSet([](auto x) { return std::cos(x); }, -6.f, 6.f, 200)};
  cos->set_data_file_name(path + "/cosine.csv");
  cos->set_column_name(0, "x");
  cos->set_column_name(1, "cos(x)");
  cos->save_data();
  auto sqrt{makeDataSet([](auto x) { return std::sqrt(x); }, 0.f, 2.f, 200)};
  sqrt->set_data_file_name(path + "/sqrt.csv");
  sqrt->set_column_name(0, "x");
  sqrt->set_column_name(1, "x^-2");
  sqrt->save_data();
  auto cube{
      makeDataSet([](auto x) { return std::pow(x, 3.f); }, -2.f, 2.f, 200)};
  cube->set_data_file_name(path + "/cube.csv");
  cube->set_column_name(0, "x");
  cube->set_column_name(1, "x^3");
  cube->save_data();
}

inline auto makeNeuralNetwork(auto numberOfInputs, auto numberOfOutputs,
                              auto layersInfo) {
  opennn::NeuralNetwork network;
  network.add_layer(new opennn::ScalingLayer{numberOfInputs});
  auto numberOfNeurons{0};
  for (size_t i = 0; i < layersInfo.size(); i++) {
    if (i % 2 != 0) {
      auto layer{new opennn::PerceptronLayer};
      layer->set_inputs_number(numberOfInputs);
      numberOfInputs = numberOfNeurons;
      layer->set_neurons_number(numberOfNeurons);
      if (layersInfo[i] == "relu") {
        layer->set_activation_function(
            opennn::PerceptronLayer::ActivationFunction::RectifiedLinear);
      }
      if (layersInfo[i] == "tanh") {
        layer->set_activation_function(
            opennn::PerceptronLayer::ActivationFunction::HyperbolicTangent);
      }
      if (layersInfo[i] == "step") {
        layer->set_activation_function(
            opennn::PerceptronLayer::ActivationFunction::Threshold);
      }
      if (layersInfo[i] == "sigmoid") {
        layer->set_activation_function(
            opennn::PerceptronLayer::ActivationFunction::HardSigmoid);
      }
      if (layersInfo[i] == "linear") {
        layer->set_activation_function(
            opennn::PerceptronLayer::ActivationFunction::Linear);
      }
      network.add_layer(layer);
    } else {
      numberOfNeurons = std::stoi(layersInfo[i]);
    }
  }
  network.add_layer(new opennn::PerceptronLayer{
      numberOfNeurons, numberOfOutputs,
      opennn::PerceptronLayer::ActivationFunction::Linear});
  network.add_layer(new opennn::UnscalingLayer{numberOfOutputs});
  network.add_layer(new opennn::BoundingLayer{numberOfOutputs});
  return network;
}

inline auto trainNeuralNetwork(auto& network, auto& dataSet, auto lossGoal,
                               auto maxEpoch, auto learningRate) {
  constexpr auto lossMethod{
      opennn::TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR};
  constexpr auto optimizationMethod{
      opennn::TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION};
  opennn::TrainingStrategy training_strategy{&network, dataSet.get()};
  training_strategy.set_loss_method(lossMethod);
  training_strategy.set_optimization_method(optimizationMethod);
  training_strategy.set_maximum_epochs_number(maxEpoch);
  training_strategy.set_loss_goal(lossGoal);
  training_strategy.get_adaptive_moment_estimation_pointer()
      ->set_initial_learning_rate(learningRate);
  training_strategy.perform_training();
}
}  // namespace

int main(int ac, char* av[]) {
  po::options_description configFileOpt{"Config file options"};
  configFileOpt.add_options()("loss-goal, lg", po::value<float>(),
                              "set loss goal")(
      "learning-rate, lr", po::value<float>(), "set learning rate")(
      "max-epoch, me", po::value<int>(), "set maximum number of epochs")(
      "layers, l", po::value<vector<string>>()->multitoken(),
      "set hidden layers info: \n"
      "ACTIVATION_FUNCTION=[relu, tanh, step, sigmoid, linear]");

  po::options_description visibleOpt{
      "Usage: approximate --data-set [PATH] --output [PATH] --layers "
      "[[NEURONS_NUMBER ACTIVATION_FUNCTION]...]"};
  visibleOpt.add_options()("help, h", "show help")(
      "config, c", po::value<string>(),
      "set configuration file with allowed parameters:\n"
      "* learning-rate\n"
      "* loss-goal\n"
      "* max-epoch\n"
      "* layers=[NUMBER_OF_NEURONS] \n"
      "* layers=[ACTIVATION_FUNCTION=[relu, tanh, step, "
      "sigmoid, linear]]...\n"
      "* layers=...")("data-set, d", po::value<string>(),
                      "set data set to approximate")(
      "output, o", po::value<string>(), "set output file to store prediction");
  po::options_description hiddenOpt{"Hidden options"};
  hiddenOpt.add_options()("generate", po::value<string>(),
                          "generate data-sets [DEST]");

  po::options_description cmdLineOpt;
  cmdLineOpt.add(hiddenOpt).add(visibleOpt);
  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, cmdLineOpt), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << visibleOpt << std::endl;
    return EXIT_SUCCESS;
  }

  if (vm.count("generate")) {
    auto path{vm["generate"].as<string>()};
    generateDataSetFiles(path);
    return EXIT_SUCCESS;
  }

  if (vm.count("config")) {
    ifstream ifs(vm["config"].as<string>());
    po::store(po::parse_config_file(ifs, configFileOpt), vm);
    notify(vm);
  } else {
    std::cout << visibleOpt << std::endl;
    return EXIT_FAILURE;
  }

  std::shared_ptr<opennn::DataSet> dataSet;
  if (vm.count("data-set")) {
    auto path{vm["data-set"].as<string>()};
    dataSet = std::make_shared<opennn::DataSet>(path, ',', true);
  } else {
    std::cout << visibleOpt << std::endl;
    return EXIT_FAILURE;
  }

  std::string predictionFile;
  if (vm.count("output")) {
    predictionFile = vm["output"].as<string>();
  } else {
    std::cout << visibleOpt << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> layersInfo;
  if (vm.count("layers")) {
    layersInfo = vm["layers"].as<vector<string>>();
  } else {
    std::cout << visibleOpt << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "layersInfo;" << std::endl;
  for (auto i : layersInfo) {
    std::cout << i << std::endl;
  }

  auto maxEpoch{1.0e4};
  if (vm.count("max-epoch")) {
    maxEpoch = vm["max-epoch"].as<int>();
  }

  auto lossGoal{1.0e-10};
  if (vm.count("loss-goal")) {
    lossGoal = vm["loss-goal"].as<float>();
  }

  auto learningRate{1.0e-2};
  if (vm.count("learning-rate")) {
    learningRate = vm["learning-rate"].as<float>();
  }

  auto numberOfInputs{dataSet->get_input_variables_number()};
  auto numberOfOutputs{dataSet->get_target_variables_number()};
  auto network{makeNeuralNetwork(numberOfInputs, numberOfOutputs, layersInfo)};

  dataSet->split_samples_random();
  trainNeuralNetwork(network, dataSet, lossGoal, maxEpoch, learningRate);
  network.save(predictionFile + ".xml");

  auto input{dataSet->get_input_data()};
  auto prediction{network.calculate_outputs(input)};
  auto numberOfSamples{dataSet->get_samples_number()};
  for (auto i = 0; i < numberOfOutputs; i++) {
    auto k{numberOfInputs + i};
    auto name{dataSet->get_columns_names()(k)};
    name.append("-prediction");
    dataSet->set_column_name(k, name);
    for (auto j = 0; j < numberOfSamples; j++) {
      dataSet->get_data_pointer()->operator()(j, k) = prediction(j, i);
    }
  }
  dataSet->set_data_file_name(predictionFile);
  dataSet->save_data();

  return EXIT_SUCCESS;
}
