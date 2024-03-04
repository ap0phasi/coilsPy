# Probabilistic Coils: Nonreal Self-Referencing Bayesian Fields
> Python package for generating and experimenting with probabilistic coils

## Table of Contents
* [Installation](#install)
* [General Info](#general-information)
* [Coil Behaviors](#coil-behaviors)
* [Physically-Based Control](#physically-based-control)
* [Pseudo Neural Network Parameterization](#pseudo-neural-network-parameterization)
* [Background](#background)
* [Usage](#usage)
<!-- * [License](#license) -->

## Installation
```

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


## Neural Network Parameterization
This package demonstrates how neural networks can be used to parameterize coils to generate dynamic fields with desired behavior.

## Background
The motivation, background, and derivation of probabilistic coils can be found here:

1. [Background]

[Background]: https://docs.google.com/document/d/e/2PACX-1vQaaN5-uBjQy8J5WLnZm3fHybOmhNjezxSUw5pn771v7gWzHI4US4KEtbtfE4dU88CzMnIz2SoLNQo2/pub

## Development Notes

### Development with Poetry
I am using Poetry for package management. This is straightforward, with the exception that if we want to use the GPU version of torch, we must do:

```
poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu121
poetry add --source pytorch torch torchvision
```
