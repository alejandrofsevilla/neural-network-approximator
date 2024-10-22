# ![LinuxWorkflow](https://github.com/alejandrofsevilla/neural-network-approximator/actions/workflows/Linux.yml/badge.svg)

# neural-network-approximator
Command line application to generate neural networks as function approximators from sets of data points.

This is an application created to experiment with the [OpenNN](https://www.opennn.net/) library for building neural networks and also, [Boost.ProgramOptions](https://www.boost.org/doc/libs/1_63_0/doc/html/program_options.html) for parsing program arguments and configuration options.

## Requirements
- C++20 compiler.
- CMake 3.22.0.
- Boost::program_options 1.74.0.

## Usage

```terminal
Usage: neural-network-approximator --config [PATH] --data-set [PATH] --output [PATH]:
  --help                show help.
  --config arg          set configuration file path.
                        Available parameters:
                        * learning-rate
                        * loss-goal
                        * max-epoch
                        * regularization=[None L1 L2]
                        * regularization-rate
                        * layers=[NUMBER_OF_NEURONS] 
                        * layers=[ACTIVATION_FUNCTION=[relu, tanh, step, 
                        sigmoid, linear]]
                        * layers=...
  --data-set arg        set data set file path.
  --output arg          set output files path.

```

## Examples
### cos(x) with 1 hidden layer = [6 tanh]:

![cosine](https://github.com/alejandrofsevilla/neural-network-approximator/assets/110661590/cc6d3412-b91f-4a25-aa64-2afd55f2c96a)

### x² with 1 hidden layer = [10 relu]:

![sqr](https://github.com/alejandrofsevilla/neural-network-approximator/assets/110661590/89a859f4-00e1-430a-84c7-a6e82ea20638)

### √x with 1 hidden layer = [10 relu]:

![sqrt](https://github.com/alejandrofsevilla/neural-network-approximator/assets/110661590/69e16565-9526-4326-a29b-5a24269164c4)
