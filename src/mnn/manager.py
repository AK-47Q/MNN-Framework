from .core import (
    NeuralNetwork,
    NetworkSpec,
    NetworkType,
    NetworkStatus
)


class NetworkManager:

    def __init__(self):

        # Network objects by their unique names.
        self.networks = {}

        # Network execution order.
        self.order = []

        # Directed communication graph.
        self.connections = {}

    # --------------------------------------------------------
    # Create
    # --------------------------------------------------------

    def create(self, spec: NetworkSpec):

        if spec.name in self.networks:
            raise ValueError("Network already exists")

        network = NeuralNetwork(spec)

        self.networks[spec.name] = network

        self.order.append(spec.name)

        self.connections[spec.name] = set()

        return network

    # --------------------------------------------------------
    # Get
    # --------------------------------------------------------

    def get(self, name):
        return self.networks[name]

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    def delete(self, name):

        if name not in self.networks:
            raise KeyError(name)

        del self.networks[name]

        # Remove from execution order.
        self.order.remove(name)

        # Remove outgoing connections.
        self.connections.pop(name, None)

        # Remove incoming connections.
        for neighbors in self.connections.values():
            neighbors.discard(name)

    # --------------------------------------------------------
    # Change network type
    # --------------------------------------------------------

    def set_type(self, name, kind):

        network = self.get(name)

        network.set_type(NetworkType(kind))

    # --------------------------------------------------------
    # Move network
    # --------------------------------------------------------

    def move(self, name, position):

        if name not in self.networks:
            raise KeyError(name)

        self.order.remove(name)

        # Clamp position to valid range.
        position = max(
            0,
            min(position, len(self.order))
        )

        self.order.insert(position, name)

    # --------------------------------------------------------
    # Modify metadata and capabilities
    # --------------------------------------------------------

    def set_metadata(self, name, key, value):

        self.get(name).spec.metadata[key] = value

    def add_capability(self, name, capability):

        self.get(name).spec.capabilities.add(capability)

    def remove_capability(self, name, capability):

        self.get(name).spec.capabilities.discard(capability)

    def set_helper_ops(self, name, operations):

        # Only known operations are accepted.
        allowed = {
            "relu",
            "tanh",
            "normalize",
            "identity"
        }

        if not set(operations).issubset(allowed):
            raise ValueError("Unknown helper operation")

        self.get(name).spec.helper_ops = list(operations)

    # --------------------------------------------------------
    # Training properties
    # --------------------------------------------------------

    def set_training(self, name, enabled):

        network = self.get(name)

        network.spec.has_optimizer = enabled

        network._configure_training()

    # --------------------------------------------------------
    # Connections
    # --------------------------------------------------------

    def connect(self, sender, receiver):

        self.get(sender)
        self.get(receiver)

        if sender == receiver:
            raise ValueError("Self connection is not allowed")

        self.connections[sender].add(receiver)

    def disconnect(self, sender, receiver):

        self.connections[sender].discard(receiver)

    def connect_all(self):

        # Fully connected directed graph.
        for sender in self.order:
            for receiver in self.order:
                if sender != receiver:
                    self.connect(sender, receiver)

    def neighbors(self, name):

        return list(self.connections[name])


# ============================================================
# High-level public API
# ============================================================

class NetworkAPI:

    def __init__(self, manager):
        self.manager = manager

    def create_network(
        self,
        name,
        kind="analytical",
        dim=10,
        experts=4,
        lr=0.001,
        has_loss=True,
        has_optimizer=True,
        capabilities=None,
        metadata=None,
        helper_ops=None
    ):

        spec = NetworkSpec(
            name=name,
            kind=NetworkType(kind),
            dim=dim,
            experts=experts,
            lr=lr,
            has_loss=has_loss,
            has_optimizer=has_optimizer,
            capabilities=set(capabilities or []),
            metadata=metadata or {},
            helper_ops=helper_ops or []
        )

        return self.manager.create(spec)

    def delete_network(self, name):
        self.manager.delete(name)

    def change_type(self, name, kind):
        self.manager.set_type(name, kind)

    def move_network(self, name, position):
        self.manager.move(name, position)

    def connect(self, sender, receiver):
        self.manager.connect(sender, receiver)

    def disconnect(self, sender, receiver):
        self.manager.disconnect(sender, receiver)

    def add_capability(self, name, capability):
        self.manager.add_capability(name, capability)

    def remove_capability(self, name, capability):
        self.manager.remove_capability(name, capability)

    def set_metadata(self, name, key, value):
        self.manager.set_metadata(name, key, value)

    def set_helper_ops(self, name, operations):
        self.manager.set_helper_ops(name, operations)

    def set_training(self, name, enabled):
        self.manager.set_training(name, enabled)

    def get_network(self, name):
        return self.manager.get(name)

    def list_networks(self):
        return list(self.manager.order)


