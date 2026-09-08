"""High-Capacity Multi-Terminal Session Network & Inter-Process Communication (IPC) Router.
Supports 500+ concurrent local and LAN networked terminal sessions with non-blocking async I/O.
Enables unique session numbering, Team Room Codes, discovery, remote command execution,
and Cross-Mode Semantic Signal Compatibility Validation (Circuit <-> Simulink <-> HDL <-> MCU).
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class SignalDomain(Enum):
    ANALOG_VOLTAGE = "ANALOG_VOLTAGE"      # Volts (Circuit Node)
    ANALOG_CURRENT = "ANALOG_CURRENT"      # Amperes (Circuit Branch)
    PHYSICAL_DYNAMIC = "PHYSICAL_DYNAMIC"  # Physical Unit (rad/s, Nm, m, deg, etc.)
    LOGIC_LEVEL = "LOGIC_LEVEL"            # 0 / 1 / High-Z (Digital Logic HDL)
    BUS_WORD = "BUS_WORD"                  # Integer bit-vector (8/16/32-bit digital bus)
    MCU_ADC = "MCU_ADC"                    # 0..3.3V / 0..5V Analog-to-Digital input
    MCU_GPIO = "MCU_GPIO"                  # 0..3.3V Digital I/O Pin
    MCU_PWM = "MCU_PWM"                    # 0..100% Duty Cycle PWM Output
    NUMERICAL_SCALAR = "NUMERICAL_SCALAR"  # Floating point matrix/scalar


@dataclass
class CrossModeValidationResult:
    """Semantic validation certificate for inter-mode cross-terminal connections."""
    COMPATIBLE: str = "COMPATIBLE"
    VALID_WITH_ADAPTER: str = "VALID_WITH_ADAPTER"
    INCOMPATIBLE: str = "INCOMPATIBLE"

    is_valid: bool = True
    status: str = "COMPATIBLE"  # "COMPATIBLE", "VALID_WITH_ADAPTER", "INCOMPATIBLE"
    source_mode: str = "CIRCUIT"
    target_mode: str = "CIRCUIT"
    source_signal: str = "V(out)"
    target_signal: str = "Vin"
    adapter_rule: Optional[str] = None
    warning_message: str = ""
    suggested_fix: str = ""

    @property
    def message(self) -> str:
        return self.warning_message or self.adapter_rule or "Semantic connection verified."

    @property
    def adapter_applied(self) -> str:
        return self.adapter_rule or "Direct 1:1 Signal Wire"

    @property
    def warnings(self) -> List[str]:
        return [self.warning_message] if self.warning_message else []


class CrossModeSignalValidator:
    """Validates whether unlike modes can safely interconnect based on physical rules and library specs."""
    COMPATIBLE: str = "COMPATIBLE"
    VALID_WITH_ADAPTER: str = "VALID_WITH_ADAPTER"
    INCOMPATIBLE: str = "INCOMPATIBLE"

    @classmethod
    def validate_connection(
        cls,
        src_mode: str,
        src_signal: str,
        dst_mode: str,
        dst_signal: str,
        src_meta: Optional[Dict[str, Any]] = None,
        dst_meta: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> CrossModeValidationResult:
        s_spec = src_meta or kwargs.get("source_spec", {})
        t_spec = dst_meta or kwargs.get("target_spec", {})
        res = cls.validate(src_mode, src_signal, dst_mode, dst_signal, s_spec, t_spec)
        if s_spec and t_spec:
            vmax = s_spec.get("voltage_max")
            adc_max = t_spec.get("adc_vmax", t_spec.get("max_voltage", 3.3))
            if vmax is not None and vmax > adc_max and ("ADC" in dst_signal.upper() or dst_mode.upper() == "EMBEDDED"):
                res.warning_message = f"Input voltage {vmax}V exceeds MCU ADC max rating {adc_max}V."
        return res

    @classmethod
    def validate(
        cls,
        source_mode: str,
        source_signal: str,
        target_mode: str,
        target_signal: str,
        source_spec: Optional[Dict[str, Any]] = None,
        target_spec: Optional[Dict[str, Any]] = None
    ) -> CrossModeValidationResult:
        s_mode = source_mode.upper().strip()
        t_mode = target_mode.upper().strip()
        s_sig = source_signal.strip()
        t_sig = target_signal.strip()

        # 1. LIKE MODES (Always directly compatible)
        if s_mode == t_mode:
            return CrossModeValidationResult(
                is_valid=True,
                status="COMPATIBLE",
                source_mode=s_mode,
                target_mode=t_mode,
                source_signal=s_sig,
                target_signal=t_sig,
                adapter_rule="Direct Passthrough (Identical Domain)"
            )

        # 2. UNLIKE MODES: CIRCUIT -> EMBEDDED (MCU)
        if s_mode == "CIRCUIT" and t_mode == "EMBEDDED":
            # Circuit voltage -> MCU ADC Pin
            if "ADC" in t_sig.upper() or "PIN" in t_sig.upper() or "AIN" in t_sig.upper():
                max_v = target_spec.get("max_voltage", 3.3) if target_spec else 3.3
                return CrossModeValidationResult(
                    is_valid=True,
                    status="VALID_WITH_ADAPTER",
                    source_mode=s_mode,
                    target_mode=t_mode,
                    source_signal=s_sig,
                    target_signal=t_sig,
                    adapter_rule=f"Analog Voltage clamped to ADC Range (0.0V to {max_v}V)",
                    warning_message=f"Ensure circuit node '{s_sig}' does not exceed MCU absolute maximum rating ({max_v}V)."
                )
            # Circuit -> MCU GPIO Digital Input
            elif "GPIO" in t_sig.upper() or "DIGITAL" in t_sig.upper():
                return CrossModeValidationResult(
                    is_valid=True,
                    status="VALID_WITH_ADAPTER",
                    source_mode=s_mode,
                    target_mode=t_mode,
                    source_signal=s_sig,
                    target_signal=t_sig,
                    adapter_rule="Schmitt-Trigger Threshold: V >= 2.0V -> Logic 1, V < 0.8V -> Logic 0"
                )

        # 3. UNLIKE MODES: EMBEDDED (MCU) -> CIRCUIT
        if s_mode == "EMBEDDED" and t_mode == "CIRCUIT":
            # MCU PWM -> Circuit Gate / Switch
            if "PWM" in s_sig.upper():
                return CrossModeValidationResult(
                    is_valid=True,
                    status="VALID_WITH_ADAPTER",
                    source_mode=s_mode,
                    target_mode=t_mode,
                    source_signal=s_sig,
                    target_signal=t_sig,
                    adapter_rule="PWM Duty Cycle mapped to 0V/3.3V pulsed gate drive signal."
                )
            # MCU GPIO Output -> Circuit node
            if "GPIO" in s_sig.upper():
                return CrossModeValidationResult(
                    is_valid=True,
                    status="VALID_WITH_ADAPTER",
                    source_mode=s_mode,
                    target_mode=t_mode,
                    source_signal=s_sig,
                    target_signal=t_sig,
                    adapter_rule="Digital State 1 -> 3.3V / State 0 -> 0.0V voltage source drive."
                )

        # 4. UNLIKE MODES: DYNAMIC (Simulink) -> CIRCUIT
        if s_mode == "DYNAMIC" and t_mode == "CIRCUIT":
            return CrossModeValidationResult(
                is_valid=True,
                status="VALID_WITH_ADAPTER",
                source_mode=s_mode,
                target_mode=t_mode,
                source_signal=s_sig,
                target_signal=t_sig,
                adapter_rule="Simulink continuous scalar mapped to Behavioral Source / Controlled Voltage (V=u)."
            )

        # 5. UNLIKE MODES: CIRCUIT -> DYNAMIC (Simulink)
        if s_mode == "CIRCUIT" and t_mode == "DYNAMIC":
            return CrossModeValidationResult(
                is_valid=True,
                status="VALID_WITH_ADAPTER",
                source_mode=s_mode,
                target_mode=t_mode,
                source_signal=s_sig,
                target_signal=t_sig,
                adapter_rule="SPICE Node Voltage / Current sampled and injected as continuous input block signal."
            )

        # 6. UNLIKE MODES: DIGITAL (HDL) -> CIRCUIT
        if s_mode == "DIGITAL" and t_mode == "CIRCUIT":
            return CrossModeValidationResult(
                is_valid=True,
                status="VALID_WITH_ADAPTER",
                source_mode=s_mode,
                target_mode=t_mode,
                source_signal=s_sig,
                target_signal=t_sig,
                adapter_rule="HDL Logic State 1 -> 5.0V (or 3.3V), State 0 -> 0.0V, High-Z -> 10Meg pull-down."
            )

        # 7. UNLIKE MODES: DIGITAL (HDL) -> EMBEDDED (MCU)
        if s_mode == "DIGITAL" and t_mode == "EMBEDDED":
            return CrossModeValidationResult(
                is_valid=True,
                status="COMPATIBLE",
                source_mode=s_mode,
                target_mode=t_mode,
                source_signal=s_sig,
                target_signal=t_sig,
                adapter_rule="Bit-level synchronous bus transfer."
            )

        # Default fallback adapter
        return CrossModeValidationResult(
            is_valid=True,
            status="VALID_WITH_ADAPTER",
            source_mode=s_mode,
            target_mode=t_mode,
            source_signal=s_sig,
            target_signal=t_sig,
            adapter_rule="Numerical scalar typecast adapter."
        )


@dataclass
class TerminalSessionInfo:
    """Metadata representing an active networked terminal session."""
    session_id: int
    client_uuid: str
    hostname: str
    ip_address: str
    port: int
    mode: str
    active_file: str
    room_code: str
    connected_at: float
    last_heartbeat: float
    tags: List[str] = field(default_factory=list)


class IPCRouter:
    """High-Scale Multi-Client Async TCP Server handling up to 500+ concurrent terminal connections.
    Supports Team Room Codes, unique session numbers, remote execution, and cross-mode signal linking.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, max_clients: int = 1000):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Session registry: session_id -> TerminalSessionInfo
        self.sessions: Dict[int, TerminalSessionInfo] = {}
        # Client mapping: writer -> session_id
        self.client_session_map: Dict[asyncio.StreamWriter, int] = {}
        # Session ID to writer
        self.session_writer_map: Dict[int, asyncio.StreamWriter] = {}
        # Room registry: room_code -> Set of session_ids
        self.rooms: Dict[str, Set[int]] = {"GLOBAL": set()}
        # Next auto-increment session number
        self._next_session_id = 1
        self._session_lock = threading.Lock()

        # Shared variable store
        self.shared_memory: Dict[str, Any] = {}
        # Topic subscriptions: topic -> set of writers
        self.subscribers: Dict[str, Set[asyncio.StreamWriter]] = {}
        # Signal links: (source_session, source_var) -> List of (target_session, target_var, adapter_info)
        self.signal_links: Dict[str, List[Dict[str, Any]]] = {}

    def start(self) -> bool:
        """Starts the router server in a dedicated background daemon thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="IPCRouterThread")
        self._thread.start()
        time.sleep(0.15)
        return True

    def stop(self) -> None:
        """Stops the router server cleanly."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_event_loop(self) -> None:
        """Asyncio event loop runner."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._start_server_coro())
            self._loop.run_forever()
        except Exception:
            pass
        finally:
            self._loop.run_until_complete(self._cleanup_coro())
            self._loop.close()

    async def _start_server_coro(self) -> None:
        """Binds TCP socket with high-concurrency backlog."""
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            backlog=self.max_clients
        )

    async def _cleanup_coro(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    def _allocate_session_id(self, requested_id: Optional[int] = None) -> int:
        """Allocates unique session number."""
        with self._session_lock:
            if requested_id is not None and requested_id > 0 and requested_id not in self.sessions:
                return requested_id
            while self._next_session_id in self.sessions:
                self._next_session_id += 1
            sid = self._next_session_id
            self._next_session_id += 1
            return sid

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handles async connection for single terminal instance."""
        sock = writer.get_extra_info("socket")
        if sock:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except Exception:
                pass

        peername = writer.get_extra_info("peername") or ("unknown", 0)
        session_id: Optional[int] = None
        current_room: str = "GLOBAL"

        try:
            while self._running:
                line = await reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue

                msg_type = msg.get("type", "")
                req_id = msg.get("id", str(uuid.uuid4()))
                payload = msg.get("payload", {})

                response: Dict[str, Any] = {"id": req_id, "status": "ok"}

                # 1. Terminal Registration & Session Numbering
                if msg_type == "register_session":
                    requested_sid = payload.get("requested_session_id")
                    session_id = self._allocate_session_id(requested_sid)
                    current_room = payload.get("room_code", "GLOBAL").upper()

                    info = TerminalSessionInfo(
                        session_id=session_id,
                        client_uuid=payload.get("client_uuid", str(uuid.uuid4())),
                        hostname=payload.get("hostname", socket.gethostname()),
                        ip_address=peername[0],
                        port=peername[1],
                        mode=payload.get("mode", "Circuit"),
                        active_file=payload.get("active_file", "untitled"),
                        room_code=current_room,
                        connected_at=time.time(),
                        last_heartbeat=time.time(),
                        tags=payload.get("tags", [])
                    )
                    self.sessions[session_id] = info
                    self.client_session_map[writer] = session_id
                    self.session_writer_map[session_id] = writer

                    if current_room not in self.rooms:
                        self.rooms[current_room] = set()
                    self.rooms[current_room].add(session_id)

                    response["session_id"] = session_id
                    response["room_code"] = current_room
                    response["session_info"] = asdict(info)
                    response["total_active_sessions"] = len(self.sessions)

                # 2. Team Room Management: Join / Switch Room
                elif msg_type == "join_room":
                    new_room = payload.get("room_code", "GLOBAL").upper()
                    if session_id and session_id in self.sessions:
                        old_room = self.sessions[session_id].room_code
                        if old_room in self.rooms and session_id in self.rooms[old_room]:
                            self.rooms[old_room].remove(session_id)
                        self.sessions[session_id].room_code = new_room
                        if new_room not in self.rooms:
                            self.rooms[new_room] = set()
                        self.rooms[new_room].add(session_id)
                        current_room = new_room
                        response["joined_room"] = new_room
                        response["room_member_count"] = len(self.rooms[new_room])

                # 3. List Sessions in Specific Team Room or Global
                elif msg_type == "list_sessions":
                    target_room = payload.get("room_code", "").upper()
                    if target_room:
                        member_ids = self.rooms.get(target_room, set())
                        session_list = [asdict(self.sessions[sid]) for sid in member_ids if sid in self.sessions]
                    else:
                        session_list = [asdict(s) for s in self.sessions.values()]
                    response["sessions"] = session_list
                    response["count"] = len(session_list)
                    response["room_code"] = target_room or "ALL"

                # 4. Remote Command Execution across Terminals
                elif msg_type == "remote_exec":
                    target_sid = payload.get("target_session_id")
                    command_str = payload.get("command", "")
                    target_writer = self.session_writer_map.get(target_sid)
                    
                    if not target_writer:
                        response["status"] = "error"
                        response["error"] = f"Session #{target_sid} is not connected to router."
                    else:
                        exec_req_id = f"exec_{uuid.uuid4().hex[:8]}"
                        exec_msg = {
                            "type": "execute_command",
                            "id": exec_req_id,
                            "source_session_id": session_id or 0,
                            "command": command_str
                        }
                        out_bytes = (json.dumps(exec_msg) + "\n").encode("utf-8")
                        target_writer.write(out_bytes)
                        await target_writer.drain()
                        response["status"] = "dispatched"
                        response["exec_req_id"] = exec_req_id

                # 5. Return Remote Command Execution Result
                elif msg_type == "exec_result":
                    origin_sid = payload.get("origin_session_id")
                    origin_writer = self.session_writer_map.get(origin_sid)
                    if origin_writer:
                        fwd = {"type": "remote_exec_complete", "payload": payload}
                        origin_writer.write((json.dumps(fwd) + "\n").encode("utf-8"))
                        await origin_writer.drain()

                # 6. Cross-Mode Semantic Signal Linking & Validation
                elif msg_type == "link_signal":
                    src_sig = payload.get("source_signal")
                    tgt_sid = payload.get("target_session_id")
                    tgt_sig = payload.get("target_signal")
                    src_mode = payload.get("source_mode", "Circuit")
                    tgt_mode = payload.get("target_mode", "Dynamic")

                    target_session = self.sessions.get(tgt_sid)
                    if target_session:
                        tgt_mode = target_session.mode

                    # Run semantic validation
                    val_res = CrossModeSignalValidator.validate(
                        source_mode=src_mode,
                        source_signal=src_sig,
                        target_mode=tgt_mode,
                        target_signal=tgt_sig
                    )

                    link_key = f"{session_id}:{src_sig}"
                    if link_key not in self.signal_links:
                        self.signal_links[link_key] = []
                    self.signal_links[link_key].append({
                        "target_session": tgt_sid,
                        "target_signal": tgt_sig,
                        "adapter_rule": val_res.adapter_rule
                    })

                    response["linked"] = f"Session #{session_id}:{src_sig} [{src_mode}] -> Session #{tgt_sid}:{tgt_sig} [{tgt_mode}]"
                    response["validation"] = asdict(val_res)

                # 7. Push Signal Value / Data Bridge
                elif msg_type == "push_signal":
                    sig_name = payload.get("signal_name")
                    val = payload.get("value")
                    link_key = f"{session_id}:{sig_name}"
                    if link_key in self.signal_links:
                        for target in self.signal_links[link_key]:
                            t_sid = target["target_session"]
                            t_sig = target["target_signal"]
                            t_writer = self.session_writer_map.get(t_sid)
                            if t_writer:
                                bridge_msg = {
                                    "type": "signal_update",
                                    "signal": t_sig,
                                    "source_session": session_id,
                                    "source_signal": sig_name,
                                    "value": val
                                }
                                t_writer.write((json.dumps(bridge_msg) + "\n").encode("utf-8"))
                                await t_writer.drain()

                # 8. Room-Scoped or Global Broadcast
                elif msg_type == "broadcast":
                    b_msg = payload.get("message", "")
                    is_room_only = payload.get("room_only", False)
                    target_room = current_room if is_room_only else None
                    await self._broadcast("room", {"from_session": session_id, "room": current_room, "message": b_msg}, target_room=target_room)
                    response["broadcast"] = "sent"

                elif msg_type == "heartbeat":
                    if session_id and session_id in self.sessions:
                        self.sessions[session_id].last_heartbeat = time.time()
                        self.sessions[session_id].mode = payload.get("mode", self.sessions[session_id].mode)
                        self.sessions[session_id].active_file = payload.get("active_file", self.sessions[session_id].active_file)
                    response["pong"] = time.time()

                out_data = (json.dumps(response) + "\n").encode("utf-8")
                writer.write(out_data)
                await writer.drain()

        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if session_id and session_id in self.sessions:
                s_room = self.sessions[session_id].room_code
                if s_room in self.rooms and session_id in self.rooms[s_room]:
                    self.rooms[s_room].remove(session_id)
                del self.sessions[session_id]
            if session_id in self.session_writer_map:
                del self.session_writer_map[session_id]
            if writer in self.client_session_map:
                del self.client_session_map[writer]

            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _broadcast(self, topic: str, payload: Any, target_room: Optional[str] = None) -> None:
        """Sends broadcast to connected terminal sessions (optionally filtered by team room)."""
        msg = json.dumps({"type": "broadcast_event", "topic": topic, "payload": payload}) + "\n"
        out_bytes = msg.encode("utf-8")

        if target_room and target_room in self.rooms:
            session_ids = self.rooms[target_room]
            writers = [self.session_writer_map[sid] for sid in session_ids if sid in self.session_writer_map]
        else:
            writers = list(self.session_writer_map.values())

        for w in writers:
            try:
                w.write(out_bytes)
            except Exception:
                pass


class IPCClient:
    """Client for terminal instance to connect to local or LAN IPCRouter.
    Supports Team Room Codes, cross-mode signal bridges, and remote execution.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.session_id: int = -1
        self.room_code: str = "GLOBAL"
        self.client_uuid = str(uuid.uuid4())
        self.is_connected = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {}
        self._command_exec_handler: Optional[Callable[[str], str]] = None
        self.linked_signals: Dict[str, Any] = {}

    def set_command_executor(self, handler: Callable[[str], str]) -> None:
        """Registers callback for executing commands sent remotely from other terminal sessions."""
        self._command_exec_handler = handler

    def connect(
        self,
        mode: str = "Circuit",
        active_file: str = "untitled",
        requested_session_id: Optional[int] = None,
        room_code: str = "GLOBAL"
    ) -> bool:
        """Connects to IPC router synchronously with timeout."""
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.5)
            s.close()
        except Exception:
            return False

        self.room_code = room_code.upper()
        self._thread = threading.Thread(
            target=self._run_client_loop,
            args=(mode, active_file, requested_session_id, self.room_code),
            daemon=True,
            name="IPCClientThread"
        )
        self._thread.start()

        for _ in range(30):
            if self.is_connected and self.session_id > 0:
                return True
            time.sleep(0.05)

        return self.is_connected

    def _run_client_loop(
        self,
        mode: str,
        active_file: str,
        requested_session_id: Optional[int],
        room_code: str
    ) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._client_coro(mode, active_file, requested_session_id, room_code))

    async def _client_coro(
        self,
        mode: str,
        active_file: str,
        requested_session_id: Optional[int],
        room_code: str
    ) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
            reg_msg = {
                "type": "register_session",
                "id": "reg_1",
                "payload": {
                    "requested_session_id": requested_session_id,
                    "client_uuid": self.client_uuid,
                    "hostname": socket.gethostname(),
                    "mode": mode,
                    "active_file": active_file,
                    "room_code": room_code
                }
            }
            self._writer.write((json.dumps(reg_msg) + "\n").encode("utf-8"))
            await self._writer.drain()

            self.is_connected = True

            while self.is_connected:
                line = await self._reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue

                if msg.get("id") == "reg_1":
                    self.session_id = msg.get("session_id", 1)
                    self.room_code = msg.get("room_code", room_code)

                if msg.get("type") == "execute_command":
                    cmd_str = msg.get("command", "")
                    src_sid = msg.get("source_session_id")
                    res_str = ""
                    if self._command_exec_handler:
                        try:
                            res_str = self._command_exec_handler(cmd_str)
                        except Exception as e:
                            res_str = f"Error: {e}"
                    else:
                        res_str = f"Session #{self.session_id} has no remote exec handler registered."

                    res_msg = {
                        "type": "exec_result",
                        "payload": {
                            "origin_session_id": src_sid,
                            "executing_session_id": self.session_id,
                            "command": cmd_str,
                            "result": str(res_str)
                        }
                    }
                    self._writer.write((json.dumps(res_msg) + "\n").encode("utf-8"))
                    await self._writer.drain()

                if msg.get("type") == "signal_update":
                    sig = msg.get("signal")
                    val = msg.get("value")
                    self.linked_signals[sig] = val

        except Exception:
            pass
        finally:
            self.is_connected = False

    def join_room_sync(self, room_code: str) -> Dict[str, Any]:
        """Joins a Team Room Code across multiple terminals/laptops."""
        if not self.is_connected:
            return {"status": "error", "error": "Not connected to IPC Router."}
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
            req = {"type": "join_room", "id": "join_room_req", "payload": {"room_code": room_code.upper()}}
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            resp = s.recv(4096).decode("utf-8")
            s.close()
            self.room_code = room_code.upper()
            return json.loads(resp)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_sessions_sync(self, room_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries router for all active terminal sessions (optionally filtered by room)."""
        if not self.is_connected:
            return []
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
            req = {"type": "list_sessions", "id": "list_req", "payload": {"room_code": room_code or ""}}
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            resp = s.recv(65536).decode("utf-8")
            s.close()
            data = json.loads(resp)
            return data.get("sessions", [])
        except Exception:
            return []

    def remote_exec_sync(self, target_session_id: int, command: str) -> Dict[str, Any]:
        """Dispatches remote command to specific session number."""
        if not self.is_connected:
            return {"status": "error", "error": "Not connected to IPC Router."}
        try:
            s = socket.create_connection((self.host, self.port), timeout=2.0)
            req = {
                "type": "remote_exec",
                "id": f"rexec_{uuid.uuid4().hex[:6]}",
                "payload": {"target_session_id": target_session_id, "command": command}
            }
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            resp = s.recv(65536).decode("utf-8")
            s.close()
            return json.loads(resp)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def link_signal_sync(
        self,
        local_signal: str,
        target_session_id: int,
        target_signal: str,
        source_mode: str = "Circuit",
        target_mode: str = "Dynamic"
    ) -> Dict[str, Any]:
        """Links local signal to a remote session's signal input with cross-mode semantic validation."""
        if not self.is_connected:
            return {"status": "error", "error": "Not connected to IPC Router."}
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
            req = {
                "type": "link_signal",
                "id": "link_1",
                "payload": {
                    "source_signal": local_signal,
                    "target_session_id": target_session_id,
                    "target_signal": target_signal,
                    "source_mode": source_mode,
                    "target_mode": target_mode
                }
            }
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            resp = s.recv(4096).decode("utf-8")
            s.close()
            return json.loads(resp)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def push_signal_sync(self, signal_name: str, value: Any) -> None:
        """Pushes current signal value across network to all linked sessions."""
        if not self.is_connected:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            req = {
                "type": "push_signal",
                "id": "push_1",
                "payload": {"signal_name": signal_name, "value": value}
            }
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            s.close()
        except Exception:
            pass

    def broadcast_sync(self, message: str, room_only: bool = False) -> None:
        """Broadcasts text or variable change to all sessions in the room."""
        if not self.is_connected:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            req = {
                "type": "broadcast",
                "id": "bcast_1",
                "payload": {"message": message, "room_only": room_only}
            }
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            s.close()
        except Exception:
            pass


# Global singleton router and client
ipc_router_instance = IPCRouter()
ipc_client_instance = IPCClient()
