"""Dashboard, Interactive Control, and Indicator blocks for Dynamic Systems."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class SliderBlock(Block):
    """Tune parameter value using sliding scale [min_val, max_val]."""

    def __init__(self, name: str, min_val: float = 0.0, max_val: float = 100.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.current_value = float(initial_value)
        self.direct_feedthrough = False

    def set_value(self, val: float) -> None:
        self.current_value = max(self.min_val, min(self.max_val, float(val)))

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.current_value
        return self.outputs


HorizontalSliderBlock = SliderBlock
VerticalSliderBlock = SliderBlock


class KnobBlock(Block):
    """Tune parameter value using rotary dial [min_val, max_val]."""

    def __init__(self, name: str, min_val: float = 0.0, max_val: float = 10.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.current_value = float(initial_value)
        self.direct_feedthrough = False

    def set_value(self, val: float) -> None:
        self.current_value = max(self.min_val, min(self.max_val, float(val)))

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.current_value
        return self.outputs


class RotarySwitchBlock(Block):
    """Switch parameter to set discrete values on dial."""

    def __init__(self, name: str, states: Sequence[float] = (0.0, 1.0, 2.0, 3.0), initial_index: int = 0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.states_list = list(states)
        self.current_idx = max(0, min(len(self.states_list) - 1, int(initial_index)))
        self.direct_feedthrough = False

    def set_index(self, idx: int) -> None:
        self.current_idx = max(0, min(len(self.states_list) - 1, int(idx)))

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.states_list[self.current_idx])
        return self.outputs


class ToggleSwitchBlock(Block):
    """Toggle parameter between two values: off_val, on_val."""

    def __init__(self, name: str, off_val: float = 0.0, on_val: float = 1.0, initial_state: bool = False):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.off_v = float(off_val)
        self.on_v = float(on_val)
        self.is_on = initial_state
        self.direct_feedthrough = False

    def toggle(self) -> None:
        self.is_on = not self.is_on

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.on_v if self.is_on else self.off_v
        return self.outputs


RockerSwitchBlock = ToggleSwitchBlock
SliderSwitchBlock = ToggleSwitchBlock


class PushButtonBlock(Block):
    """Change parameter value while button is pressed: pressed_value, released_value."""

    def __init__(self, name: str, pressed_value: float = 1.0, released_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.p_val = float(pressed_value)
        self.r_val = float(released_value)
        self.is_pressed = False
        self.direct_feedthrough = False

    def press(self) -> None:
        self.is_pressed = True

    def release(self) -> None:
        self.is_pressed = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.p_val if self.is_pressed else self.r_val
        return self.outputs


class CallbackButtonBlock(Block):
    """Execute callback function on button trigger."""

    def __init__(self, name: str, callback: Optional[Callable[[], None]] = None):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.cb = callback
        self.fired = False
        self.direct_feedthrough = False

    def click(self) -> None:
        if self.cb:
            self.cb()
        self.fired = True

    def compute_output(self, t: float) -> List[float]:
        val = 1.0 if self.fired else 0.0
        self.fired = False
        self.outputs[0] = val
        return self.outputs


class CheckBoxBlock(Block):
    """Select binary parameter value (0.0 or 1.0)."""

    def __init__(self, name: str, initial_checked: bool = False):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.checked = initial_checked
        self.direct_feedthrough = False

    def set_checked(self, checked: bool) -> None:
        self.checked = bool(checked)

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = 1.0 if self.checked else 0.0
        return self.outputs


class ComboBoxBlock(Block):
    """Select parameter value from dropdown list."""

    def __init__(self, name: str, options: Sequence[Any] = (0.0, 1.0, 2.0), initial_index: int = 0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.opts = list(options)
        self.selected_idx = max(0, min(len(self.opts) - 1, int(initial_index)))
        self.direct_feedthrough = False

    def select(self, index: int) -> None:
        self.selected_idx = max(0, min(len(self.opts) - 1, int(index)))

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.opts[self.selected_idx]
        return self.outputs


RadioButtonBlock = ComboBoxBlock


class EditBlock(Block):
    """Enter new value for parameter dynamically during simulation."""

    def __init__(self, name: str, initial_value: Any = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.val = initial_value
        self.direct_feedthrough = False

    def set_text(self, text: str) -> None:
        try:
            self.val = float(text)
        except ValueError:
            self.val = text

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.val
        return self.outputs


CustomEditBoxBlock = EditBlock


class GaugeBlock(Block):
    """Display signal value on circular/linear scale."""

    def __init__(self, name: str, min_val: float = 0.0, max_val: float = 100.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.min_v = float(min_val)
        self.max_v = float(max_val)
        self.current_reading = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.current_reading = max(self.min_v, min(self.max_v, u))
        self.outputs[0] = self.current_reading
        return self.outputs


CircularGaugeBlock = GaugeBlock
HalfGaugeBlock = GaugeBlock
LinearGaugeBlock = GaugeBlock
QuarterGaugeBlock = GaugeBlock
HorizontalGaugeBlock = GaugeBlock
VerticalGaugeBlock = GaugeBlock


class LampBlock(Block):
    """Display color/state that reflects signal value (e.g. thresholds for Green, Yellow, Red)."""

    def __init__(
        self,
        name: str,
        thresholds: Sequence[float] = (0.0, 50.0, 80.0),
        colors: Sequence[str] = ("green", "yellow", "red")
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.thresh = list(thresholds)
        self.colors = list(colors)
        self.current_color = self.colors[0]
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        u = float(self.inputs[0])
        col = self.colors[0]
        for i, th in enumerate(self.thresh):
            if u >= th and i < len(self.colors):
                col = self.colors[i]
        self.current_color = col
        self.outputs[0] = col
        return self.outputs


class MultiStateImageBlock(Block):
    """Display image/icon index reflecting input value."""

    def __init__(self, name: str, num_states: int = 4):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.num_states = num_states
        self.current_state_idx = 0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[int]:
        u = int(round(float(self.inputs[0])))
        self.current_state_idx = max(0, min(self.num_states - 1, u))
        self.outputs[0] = self.current_state_idx
        return self.outputs


class DashboardDisplayBlock(Block):
    """Display signal value in box during simulation."""

    def __init__(self, name: str, format_spec: str = "{:.3f}"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fmt = format_spec
        self.display_str = ""
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        val = self.inputs[0]
        try:
            self.display_str = self.fmt.format(val)
        except Exception:
            self.display_str = str(val)
        self.outputs[0] = self.display_str
        return self.outputs


class DashboardScopeBlock(Block):
    """Trace signals on scope display during simulation."""

    def __init__(self, name: str, buffer_len: int = 100):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.buf_len = buffer_len
        self.history: List[float] = []
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.history.append(float(self.inputs[0]))
        if len(self.history) > self.buf_len:
            self.history.pop(0)
        self.outputs[0] = list(self.history)
        return self.outputs
