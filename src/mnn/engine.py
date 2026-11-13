from concurrent.futures import ThreadPoolExecutor

import torch

from .core import NetworkStatus
from .communication import MessageBus


class MNNEngine:

    def __init__(
        self,
        manager,
        workers=8,
        rounds=10
    ):

        self.manager = manager

        self.workers = workers
        self.rounds = rounds

        self.bus = MessageBus()

    def _run_one(self, name, context, inbox):

        network = self.manager.get(name)

        return network.compute(context, inbox)

    def run(self, context):

        # Context shape: [batch, shared_dimension].
        if context.ndim != 2:
            raise ValueError("Context must have shape [batch, dim]")

        # Every network must use the same message dimension.
        for name in self.manager.order:
            network = self.manager.get(name)

            if network.spec.dim != context.shape[-1]:
                raise ValueError(
                    f"Dimension mismatch in {name}"
                )

        # Initial states.
        states = {}

        for name in self.manager.order:
            states[name] = torch.zeros_like(context)

        # Each round uses a snapshot of previous messages.
        # Messages generated in round r are delivered in r+1.
        with ThreadPoolExecutor(
            max_workers=self.workers
        ) as pool:

            for round_id in range(self.rounds):

                names = [
                    name for name in self.manager.order
                    if self.manager.get(name).status
                    != NetworkStatus.DISABLED
                ]

                futures = {}

                # Read incoming messages before launching workers.
                inboxes = {}

                for name in names:

                    messages = self.bus.receive(
                        name,
                        round_id
                    )

                    inboxes[name] = [
                        message.payload
                        for message in messages
                    ]

                # Run all networks in parallel.
                for name in names:

                    futures[name] = pool.submit(
                        self._run_one,
                        name,
                        context,
                        inboxes[name]
                    )

                # Barrier: wait for every network.
                results = {}

                for name, future in futures.items():
                    results[name] = future.result()

                # Update states only after the barrier.
                states.update({
                    name: result[0]
                    for name, result in results.items()
                })

                # Send messages for the next round.
                for sender, result in results.items():

                    output = result[0]

                    for receiver in self.manager.neighbors(sender):

                        if receiver not in names:
                            continue

                        self.bus.send(
                            sender=sender,
                            receiver=receiver,
                            round_id=round_id + 1,
                            payload=output,
                            message_type="feature"
                        )

        # Final outputs are returned only after all rounds.
        self.bus.clear()

        # Aggregate predictions from non-helper networks.
        active_outputs = [
            states[name]
            for name in self.manager.order
            if (
                self.manager.get(name).kind.value != "helper"
                and self.manager.get(name).status
                != NetworkStatus.DISABLED
            )
        ]

        if not active_outputs:
            active_outputs = [
                states[name]
                for name in self.manager.order
                if self.manager.get(name).status
                != NetworkStatus.DISABLED
            ]

        if not active_outputs:
            raise RuntimeError("No available networks")

        final_output = torch.stack(
            active_outputs
        ).mean(dim=0)

        return {
            "output": final_output,
            "networks": states
        }


