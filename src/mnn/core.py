from enum import Enum
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# Netwoek types

class NetworkType(str, Enum):
    ANALYTICAL = "analytical"
    EXTERNAL = "external"
    HELPER = "helper"


class NetworkStatus(str, Enum):
    ACTIVE = "active"
    PROBATION = "probation"
    DISABLED = "disabled"



# Network configuration

@dataclass
class NetworkSpec:
    # Unique network name.
    name: str

    # Functional role of this network.
    kind: NetworkType = NetworkType.ANALYTICAL

    # Input and hidden dimensions.
    # This framework uses a shared message dimension.
    dim: int = 10

    # Number of experts inside this network.
    experts: int = 4

    # Learning rate for trainable networks.
    lr: float = 0.001

    # Whether this network has an independent loss.
    has_loss: bool = True

    # Whether it can own an optimizer.
    has_optimizer: bool = True

    # Capabilities describe what this network can do.
    capabilities: set = field(default_factory=set)

    # Extra user-defined metadata.
    metadata: dict = field(default_factory=dict)

    # Operations allowed when this network is a helper.
    helper_ops: list = field(default_factory=list)


# ============================================================
# Expert
# ============================================================

class Expert(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # Three weight matrices.
        # Each matrix has dim * dim weights.
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

    def forward(self, x):

        # First transformation.
        x = torch.tanh(self.fc1(x))

        # Second transformation.
        x = torch.tanh(self.fc2(x))

        # Third transformation.
        x = self.fc3(x)

        return x


# ============================================================
# Mixture of Experts
# ============================================================

class MoE(nn.Module):

    def __init__(self, dim, expert_count):
        super().__init__()

        # Independent experts.
        self.experts = nn.ModuleList([
            Expert(dim) for _ in range(expert_count)
        ])

        # Gate chooses how much each expert contributes.
        self.gate = nn.Linear(dim, expert_count)

    def forward(self, x):

        # Calculate routing probabilities.
        weights = F.softmax(self.gate(x), dim=-1)

        # Run all experts.
        outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=-2
        )

        # Weighted aggregation.
        result = (
            outputs * weights.unsqueeze(-1)
        ).sum(dim=-2)

        return result, weights


# ============================================================
# Neural Network
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, spec: NetworkSpec):
        super().__init__()

        self.spec = spec

        self.name = spec.name
        self.kind = spec.kind

        self.status = NetworkStatus.ACTIVE

        # Every network owns its own MoE.
        self.moe = MoE(spec.dim, spec.experts)

        # Last output is useful for debugging.
        self.last_output = None

        # Helpers do not train independently.
        self.optimizer = None

        self._configure_training()

    def _configure_training(self):

        # Helpers cannot own independent optimizers.
        if self.kind == NetworkType.HELPER:
            self.optimizer = None
            return

        if self.spec.has_optimizer:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.spec.lr
            )

    def set_type(self, kind):

        # Update network role.
        self.kind = NetworkType(kind)
        self.spec.kind = self.kind

        # Helpers stop independent training.
        self._configure_training()

    def helper_forward(self, x):

        # Helpers execute only registered operations.
        # They cannot execute arbitrary code from messages.

        for operation in self.spec.helper_ops:

            if operation == "relu":
                x = F.relu(x)

            elif operation == "tanh":
                x = torch.tanh(x)

            elif operation == "normalize":
                x = F.normalize(x, dim=-1)

            elif operation == "identity":
                pass

        return x

    def forward(self, x):

        # Disabled networks cannot participate.
        if self.status == NetworkStatus.DISABLED:
            raise RuntimeError(
                f"Network {self.name} is disabled"
            )

        # Helper networks use registered operations.
        if self.kind == NetworkType.HELPER:
            result = self.helper_forward(x)

            self.last_output = result.detach()

            return result, None

        # Normal networks use their own MoE.
        result, gate = self.moe(x)

        self.last_output = result.detach()

        return result, gate

    def compute(self, context, inbox):

        # Combine incoming messages.
        if inbox:
            incoming = torch.stack(inbox).mean(dim=0)
        else:
            incoming = torch.zeros_like(context)

        # The network receives both its context and messages.
        x = context + incoming

        # Inference only.
        with torch.no_grad():
            result, gate = self.forward(x)

        return result.detach(), gate


# ============================================================
# Message
# ============================================================

@dataclass
class Message:

    sender: str
    receiver: str

    # Message delivery round.
    round_id: int

    # Message category.
    message_type: str

    # Tensor payload.
    payload: torch.Tensor

    # Optional confidence.
    confidence: float = 1.0

    # Additional metadata.
    metadata: dict = field(default_factory=dict)


