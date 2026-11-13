import torch

from mnn import MNN


# Create the system.
mnn = MNN(
    workers=8,
    rounds=10
)


# Create networks with different roles.
mnn.networks.create_network(
    name="analytical_1",
    kind="analytical",
    dim=10,
    experts=4,
    capabilities=["character_analysis"]
)

mnn.networks.create_network(
    name="analytical_2",
    kind="analytical",
    dim=10,
    experts=4,
    capabilities=["spelling_analysis"]
)

mnn.networks.create_network(
    name="external_1",
    kind="external",
    dim=10,
    experts=4,
    capabilities=["word_prediction"]
)

mnn.networks.create_network(
    name="external_2",
    kind="external",
    dim=10,
    experts=4,
    capabilities=["error_prediction"]
)


# Connect every network to every other network.
mnn.manager.connect_all()


# Create a helper.
mnn.networks.create_network(
    name="helper_1",
    kind="helper",
    dim=10,
    experts=2,
    has_loss=False,
    has_optimizer=False,
    helper_ops=["normalize", "tanh"]
)


# Demonstrate network management.
mnn.networks.move_network(
    "helper_1",
    0
)

mnn.networks.add_capability(
    "helper_1",
    "feature_transform"
)

mnn.networks.set_metadata(
    "analytical_1",
    "description",
    "Analyzes character patterns"
)


# Run the system.
context = torch.randn(4, 10)

result = mnn.run(context)

print("Final output:")
print(result["output"])


# External evaluation.
# These metrics are examples only.
# Real metrics should come from validation data.
report = mnn.evaluation.evaluate(
    "analytical_1",
    {
        "loss": 0.5,
        "accuracy": 0.8
    }
)

print("Evaluation:")
print(report)


# Modify a network.
mnn.modification.set_learning_rate(
    "external_1",
    0.0005
)


# Change its role.
mnn.networks.change_type(
    "external_2",
    "helper"
)

mnn.networks.set_helper_ops(
    "external_2",
    ["relu", "normalize"]
)


# List networks.
print("Network order:")
print(mnn.networks.list_networks())


# Delete a network.
# mnn.networks.delete_network("helper_1")

