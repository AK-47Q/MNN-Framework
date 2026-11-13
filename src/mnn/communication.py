from collections import defaultdict

from .core import Message


class MessageBus:

    def __init__(self):

        # Messages are grouped by delivery round and receiver.
        self.messages = defaultdict(list)

    def send(
        self,
        sender,
        receiver,
        round_id,
        payload,
        message_type="feature",
        confidence=1.0,
        metadata=None
    ):

        # Store a detached copy to avoid accidental autograd
        # graph sharing between independent networks.
        message = Message(
            sender=sender,
            receiver=receiver,
            round_id=round_id,
            message_type=message_type,
            payload=payload.detach().clone(),
            confidence=confidence,
            metadata=metadata or {}
        )

        self.messages[(round_id, receiver)].append(message)

    def receive(self, receiver, round_id):

        key = (round_id, receiver)

        result = self.messages.pop(key, [])

        return result

    def clear(self):
        self.messages.clear()


