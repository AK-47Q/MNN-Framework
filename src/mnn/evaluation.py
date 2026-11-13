from collections import defaultdict

from .core import NetworkType, NetworkStatus


class EvaluationAPI:

    def __init__(
        self,
        manager,
        loss_limit=2.0,
        accuracy_limit=0.3,
        probation_patience=3,
        failure_patience=5
    ):

        self.manager = manager

        self.loss_limit = loss_limit
        self.accuracy_limit = accuracy_limit

        self.probation_patience = probation_patience
        self.failure_patience = failure_patience

        # Consecutive bad evaluations.
        self.bad_counts = defaultdict(int)

        # Evaluation history.
        self.history = defaultdict(list)

    def evaluate(self, name, metrics):

        network = self.manager.get(name)

        # Metrics must come from an external evaluator
        # using independent validation data.
        loss = float(metrics["loss"])

        accuracy = metrics.get("accuracy")
        accuracy = (
            None if accuracy is None else float(accuracy)
        )

        bad = loss > self.loss_limit

        if accuracy is not None:
            bad = bad or accuracy < self.accuracy_limit

        record = {
            "loss": loss,
            "accuracy": accuracy,
            "bad": bad
        }

        self.history[name].append(record)

        if bad:
            self.bad_counts[name] += 1
        else:
            self.bad_counts[name] = 0

            if network.status == NetworkStatus.PROBATION:
                network.status = NetworkStatus.ACTIVE

        count = self.bad_counts[name]

        # First stage: probation.
        if count >= self.probation_patience:
            network.status = NetworkStatus.PROBATION

        # Second stage: convert to helper.
        if count >= self.failure_patience:

            # Preserve the network's original weights.
            # Training is disabled by changing its type.
            network.set_type(NetworkType.HELPER)

        return {
            "network": name,
            "metrics": record,
            "bad_count": count,
            "status": network.status.value,
            "type": network.kind.value
        }

    def promote(self, name, kind="analytical"):

        # Manual rehabilitation.
        network = self.manager.get(name)

        network.set_type(NetworkType(kind))
        network.status = NetworkStatus.ACTIVE

        self.bad_counts[name] = 0

    def disable(self, name):

        network = self.manager.get(name)

        network.status = NetworkStatus.DISABLED

    def enable(self, name):

        network = self.manager.get(name)

        network.status = NetworkStatus.ACTIVE

    def report(self, name):

        return list(self.history[name])


# ============================================================
# Modification API
# ============================================================

class ModificationAPI:

    def __init__(self, manager):
        self.manager = manager

    def set_learning_rate(self, name, lr):

        if lr <= 0:
            raise ValueError("Learning rate must be positive")

        network = self.manager.get(name)

        network.spec.lr = lr

        # Recreate optimizer with the new learning rate.
        if network.kind != NetworkType.HELPER:
            network.optimizer = __import__(
                "torch"
            ).optim.AdamW(
                network.parameters(),
                lr=lr
            )

    def set_expert_count(self, name, count):

        # Replacing MoE changes the network architecture.
        # Existing expert weights are not preserved here.
        import torch.nn as nn
        from .core import MoE

        if count < 1:
            raise ValueError("At least one expert is required")

        network = self.manager.get(name)

        network.moe = MoE(
            network.spec.dim,
            count
        )

        network.spec.experts = count

        network._configure_training()

    def reset_network(self, name):

        # Reinitialize parameters.
        network = self.manager.get(name)

        def reset(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        network.apply(reset)

        network._configure_training()


