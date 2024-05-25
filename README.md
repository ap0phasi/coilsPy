# Probabilistic Coils: Nonreal Self-Referencing Bayesian Fields
> Python package for generating and experimenting with probabilistic coils

## Table of Contents
* [Installation](#install)
* [General Info](#general-information)
* [Coil Behaviors](#coil-behaviors)
* [Physically-Based Control](#physically-based-control)
* [Background](#background)
* [Neural Coil](#neural-coil)
* [Coil Normalization](#coil-normalization)
* [Development](#development)
* [License](#license) -->

## Installation

Install with
```
pip install coilspy
```

## General Information
Probabilistic coils are systems of interacting, conserved, nonreal-valued Bayesian fields where the conditionals
are themselves dependent on all state and conditional probabilities, thus making coils
self-referencing. 

Probabilistic coils are inspired by the need for a mathematical framework to describe
dynamic, interconnected, non-hierarchical systems. By using conserved Bayesian fields,
we can describe the flow of discrete state probabilities. By making these self-referencing,
we can describe interdependent probabilistic flows. The generalization into complex and quaternionic
number systems offers wider extensibility. 

## Coil Behaviors 
Probabilistic coils exhibit a number of interesting behaviors. One key behavior is sustained 
aperiodic oscillation. As a result, many coils exhibit chaos. 

It should be emphasized that coils behave deterministically, thus irregular phenomenon is a result of
interconnectedness. 


## Physically-Based Control
Coils can be constructed with a variety of physically-based constraints. For example, locality
can be enforced, preventing the flow of probability to non-neighboring states. Inertial biases can
be imposed, decreasing the flow of probability out of a state. 

Locality constraints can also be used to sever coils, resulting in separate interacting conserved subcoils.
Parameter symmetry can be used to formulate coils with identical parameterizations.

## Background
The motivation, background, and derivation of probabilistic coils can be found here:

1. [Background]

[Background]: https://docs.google.com/document/d/e/2PACX-1vQaaN5-uBjQy8J5WLnZm3fHybOmhNjezxSUw5pn771v7gWzHI4US4KEtbtfE4dU88CzMnIz2SoLNQo2/pub

## Neural Coil Layer

This package also includes an experimental ```torch``` layer meant to emulate coil behaviour. This can be imported in the following manner:

```
from coilspy.neuralcoil import NeuralCoilLayer

n_features = 16
batch, length, dim = 13, 64, n_features
x = torch.randn(batch, length, dim)
model = NeuralCoilLayer(
    n_features = n_features,
    n_batch=batch
)
y = model(x)
```
While there is no code overlap, I have to give a shoutout to [Mamba](https://github.com/state-spaces/mamba). I was working on the complex coil generator portion of this package with no intent in trying to turn this into a neural network layer, but the handling of state spaces in that package was an inspiration for this development. 

The NeuralCoil layer is meant to process sequence data that is normalized (each step summing to 1). Please use the [Coil Normalization](#coil-normalization) to help with this. 

## Coil Normalization
Because coils are meant to describe the flow of probability, and many sequences typically are not directly formulated in the terms of probabilities, I have developed a method to translate timeseries into probabilistic flows. Simply, if consider a signal increasing by some maximum amount in a single step as one state, and the signal decreasing by some maximum amount in a step as our other state, then all dynamics can be described as some weighted combination of these states. 

To use coil normalization

```
from coilspy.normalization import CoilNormalizer

coilnormer = CoilNormalizer()

coilnormed_df = coilnormer.normalize(df)
```

This expands the feature dimension of the timeseries, so a 14 features timeseries would be normalized to a 28 feature timeseries, as each of the original 14 features is described as some combination of an "up state" and "down state".

## Examples

Some helpful notebooks can be found here:
1. [Complex Coils From Scratch] steps through the process of creating the complex coil generators from principles described in the doc. 
2. [Complex Coil Package Setup] shows how the complex coils functions can be practically used.
3. [Neural Coils and Coil Normalization] shows the usage of coil normalization and neural coil layers on the [Jena climate dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate). 

[Complex Coils From Scratch]: https://github.com/ap0phasi/coilsPy/blob/main/dev/coils_from_scratch/coils_from_scratch_13.ipynb
[Complex Coil Package Setup]: https://github.com/ap0phasi/coilsPy/blob/main/tests/coil_test.ipynb
[Neural Coils and Coil Normalization]: https://github.com/ap0phasi/coilsPy/blob/main/tests/coilnorm_test.ipynb

## Development Notes

This is still very much under development, and everything is in the purely experimental phase. I hope to start linking this into my other projects, such as my [Scedastic Surrogate Swarm Optimizer](https://github.com/ap0phasi/ScedasticSurrogateSwarmPy) and the [Cerberus Timeseries Forecasting Project](https://github.com/ap0phasi/cerberusPy).

_Last Updated: 2023-12-27_

### Development with Poetry
I am using Poetry for package management. This is straightforward, with the exception that if we want to use the GPU version of torch, we must do:

```
poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu121
poetry add --source pytorch torch torchvision
```
