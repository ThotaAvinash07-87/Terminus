"""Messages, Queues, Event Schedulers, and Sequence Viewers for Dynamic Systems."""

from __future__ import annotations
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class QueueBlock(Block):
    """Enqueue messages and entities with FIFO or LIFO discipline:
    Input 0: message data
    Input 1: push trigger (>0)
    Input 2: pop trigger (>0)
    Outputs: [popped_msg, current_queue_length]
    """

    def __init__(self, name: str, capacity: int = 100, lifo: bool = False):
        super().__init__(name, num_inputs=3, num_outputs=2)
        self.capacity = capacity
        self.lifo = lifo
        self.queue: deque = deque(maxlen=capacity)
        self.prev_push = 0.0
        self.prev_pop = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        msg = self.inputs[0]
        push = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        pop = float(self.inputs[2]) if len(self.inputs) > 2 else 0.0

        if push > 0.0 and self.prev_push <= 0.0:
            self.queue.append(msg)

        popped = 0.0
        if pop > 0.0 and self.prev_pop <= 0.0 and len(self.queue) > 0:
            popped = self.queue.pop() if self.lifo else self.queue.popleft()

        self.prev_push = push
        self.prev_pop = pop

        self.outputs[0] = popped
        self.outputs[1] = float(len(self.queue))
        return self.outputs

    def reset_states(self) -> None:
        self.queue.clear()
        self.prev_push = 0.0
        self.prev_pop = 0.0


class SendBlock(Block):
    """Create and send message payload to named channel."""

    _CHANNELS: Dict[str, deque] = {}

    def __init__(self, name: str, channel_name: str = "chan1"):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.chan = channel_name.strip()
        if self.chan not in self._CHANNELS:
            self._CHANNELS[self.chan] = deque()
        self.prev_send = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        msg = self.inputs[0]
        send_trigger = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0

        sent = 0.0
        if send_trigger > 0.0 and self.prev_send <= 0.0:
            self._CHANNELS[self.chan].append(msg)
            sent = 1.0
        self.prev_send = send_trigger
        self.outputs[0] = sent
        return self.outputs


class ReceiveBlock(Block):
    """Receive messages from named channel."""

    def __init__(self, name: str, channel_name: str = "chan1"):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.chan = channel_name.strip()
        self.prev_req = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        req = float(self.inputs[0]) if len(self.inputs) > 0 else 1.0
        chan_queue = SendBlock._CHANNELS.get(self.chan, deque())

        msg = 0.0
        has_msg = 0.0
        if req > 0.0 and self.prev_req <= 0.0 and len(chan_queue) > 0:
            msg = chan_queue.popleft()
            has_msg = 1.0
        self.prev_req = req

        self.outputs[0] = msg
        self.outputs[1] = has_msg
        return self.outputs


class MessageMergeBlock(Block):
    """Combine message paths into one."""

    def __init__(self, name: str, num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        for msg in self.inputs:
            if msg is not None and (not isinstance(msg, (int, float)) or abs(msg) > 1e-12):
                self.outputs[0] = msg
                return self.outputs
        self.outputs[0] = 0.0
        return self.outputs


class MessageTriggeredSubsystemBlock(Block):
    """Subsystem whose execution is controlled by message arrival."""

    def __init__(self, name: str, action_fn: Callable[[Any], Any]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.action = action_fn
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        msg = self.inputs[0]
        if msg is not None and (not isinstance(msg, (int, float)) or abs(msg) > 1e-12):
            self.outputs[0] = self.action(msg)
        return self.outputs


class HitSchedulerBlock(Block):
    """Schedule major time steps for variable-step solvers."""

    def __init__(self, name: str, hit_times: Sequence[float] = (1.0, 2.0, 5.0)):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.hits = sorted(list(hit_times))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        is_hit = 1.0 if any(abs(t - ht) < 1e-6 for ht in self.hits) else 0.0
        self.outputs[0] = is_hit
        return self.outputs


class SequenceViewerBlock(Block):
    """Display and record sequence of messages, transitions, and event timestamps."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.events: List[Tuple[float, Any]] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        msg = self.inputs[0]
        if msg is not None and (not isinstance(msg, (int, float)) or abs(msg) > 1e-12):
            self.events.append((t, msg))
        return []

    def reset_states(self) -> None:
        self.events.clear()
