# ![LinuxWorkflow](https://github.com/alejandrofsevilla/neural-network-approximator/actions/workflows/Linux.yml/badge.svg)
# neural-network-approximator
Command line application using [Opennn](https://www.opennn.net/) library to generate a neural network as a function approximator of a set of data points.

## Requirements
- C++20 compiler.
- CMake 3.22.0.
- Boost::program_options 1.74.0.

## Usage

```shell
approximate --data-set [PATH] --output [PATH] --layers [[NEURONS_NUMBER ACTIVATION_FUNCTION]...]
```
where
```shell
ACTIVATION_FUNCTION = [relu, tanh, step, sigmoid, linear]
```

## Examples
