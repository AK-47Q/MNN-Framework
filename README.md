# MNN Framework

A modular multi-neural-network framework.

## Features

- Create and delete networks.
- Change network types.
- Modify network metadata and capabilities.
- Move networks in execution order.
- Directed network connections.
- ThreadPoolExecutor based parallel execution.
- Synchronous communication rounds.
- Message bus with typed messages.
- Mixture of Experts inside every network.
- External Evaluation API.
- Network probation and helper conversion.
- Learning-rate modification.
- Expert-count modification.
- Network reinitialization.

## Installation

```bash
python install.py
```

## Run example

```bash
python examples/basic.py
```

## Network types

- ANALYTICAL
- EXTERNAL
- HELPER

## Important architecture notes

The execution engine uses synchronous communication rounds.

Messages generated in round N are delivered in round N+1.

The engine uses a thread pool. Each network computes independently
inside its worker, then a barrier synchronizes the round.

The evaluation API is external to neural-network inference.

Evaluation metrics must come from an independent validation process.

A helper does not train independently. It can execute only registered
operations.

The current implementation uses a shared message dimension across
networks. Different dimensions require explicit message adapters.

The current MoE implementation evaluates every expert and then
aggregates their outputs using a learned gate. It is not sparse
top-k routing yet.
