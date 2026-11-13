import torch

from mnn import MNN


def test_network_creation():

    system = MNN(workers=2, rounds=2)

    system.networks.create_network(
        "a",
        kind="analytical",
        dim=10
    )

    system.networks.create_network(
        "b",
        kind="external",
        dim=10
    )

    system.networks.connect("a", "b")
    system.networks.connect("b", "a")

    assert len(system.networks.list_networks()) == 2

    result = system.run(torch.randn(2, 10))

    assert result["output"].shape == (2, 10)


def test_network_modification():

    system = MNN()

    system.networks.create_network(
        "a",
        dim=10
    )

    system.networks.change_type("a", "helper")

    assert (
        system.networks.get_network("a").kind.value
        == "helper"
    )

    system.networks.delete_network("a")

    assert system.networks.list_networks() == []


