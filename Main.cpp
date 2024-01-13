#include <opennn/data_set.h>
#include <opennn/neural_network.h>
#include <opennn/training_strategy.h>

#include <boost/program_options.hpp>

namespace {

namespace po = boost::program_options;

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
                               auto maxEpoch, auto learningRate,
                               auto regularization, auto regularizationRate) {
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
  training_strategy.get_loss_index_pointer()->set_regularization_method(
      regularization);
  training_strategy.get_loss_index_pointer()->set_regularization_weight(
      regularizationRate);
  training_strategy.perform_training();
}
}  // namespace

int main(int ac, char* av[]) {
  po::options_description configFileOpt{"Config file options"};
  configFileOpt.add_options()("loss-goal, lg", po::value<float>())(
      "learning-rate, lr", po::value<float>())("max-epoch, me",
                                               po::value<int>())(
      "regularization, r", po::value<string>())("regularization-rate, rr",
                                                po::value<float>())(
      "layers, l", po::value<vector<string>>()->multitoken());

  po::options_description visibleOpt{
      "Usage: neural-network-approximator --config [PATH] --data-set [PATH] "
      "--output [PATH]"};
  visibleOpt.add_options()("help, h", "show help.")(
      "config, c", po::value<string>(),
      "set configuration file path.\n"
      "Available parameters:\n"
      "* learning-rate\n"
      "* loss-goal\n"
      "* max-epoch\n"
      "* regularization=[None L1 L2]\n"
      "* regularization-rate\n"
      "* layers=[NUMBER_OF_NEURONS] \n"
      "* layers=[ACTIVATION_FUNCTION=[relu, tanh, step, sigmoid, linear]]\n"
      "* layers=...")("data-set, d", po::value<string>(),
                      "set data set file path.")(
      "output, o", po::value<string>(), "set output files path.");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, visibleOpt), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << visibleOpt << std::endl;
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

  auto maxEpoch{10000};
  if (vm.count("max-epoch")) {
    maxEpoch = vm["max-epoch"].as<int>();
  }

  auto lossGoal{1.0e-3};
  if (vm.count("loss-goal")) {
    lossGoal = vm["loss-goal"].as<float>();
  }

  auto learningRate{0.03};
  if (vm.count("learning-rate")) {
    learningRate = vm["learning-rate"].as<float>();
  }

  auto regularization{
      opennn::LossIndex::RegularizationMethod::NoRegularization};
  if (vm.count("regularization")) {
    auto reg = vm["regularization"].as<string>();
    if (reg == "L1") {
      regularization = opennn::LossIndex::RegularizationMethod::L1;
    } else if (reg == "L2") {
      regularization = opennn::LossIndex::RegularizationMethod::L2;
    } else if (reg != "None") {
      std::cout << visibleOpt << std::endl;
      return EXIT_FAILURE;
    }
  }

  auto regularizationRate{0.f};
  if (vm.count("regularization-rate")) {
    regularizationRate = vm["regularization-rate"].as<float>();
  }

  auto numberOfInputs{dataSet->get_input_variables_number()};
  auto numberOfOutputs{dataSet->get_target_variables_number()};
  auto network{makeNeuralNetwork(numberOfInputs, numberOfOutputs, layersInfo)};

  dataSet->split_samples_random();
  trainNeuralNetwork(network, dataSet, lossGoal, maxEpoch, learningRate,
                     regularization, regularizationRate);
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
