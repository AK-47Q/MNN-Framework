from .manager import NetworkManager, NetworkAPI
from .evaluation import EvaluationAPI, ModificationAPI
from .engine import MNNEngine


class MNN:

    def __init__(
        self,
        workers=8,
        rounds=10
    ):

        # central registry.
        self.manager = NetworkManager()

        # public network management API
        self.networks = NetworkAPI(self.manager)

        # external evaluation API
        self.evaluation = EvaluationAPI(self.manager)

        # External modification API
        self.modification = ModificationAPI(self.manager)

        # cmon (communication, damn its so long (not that weirdo)) and threaded execution engine
        self.engine = MNNEngine(
            self.manager,
            workers=workers,
            rounds=rounds
        )

    def run(self, context):
        return self.engine.run(context)


