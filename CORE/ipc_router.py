"""High-Capacity Multi-Terminal Session Network & Inter-Process Communication (IPC) Router.
Supports 500+ concurrent local and LAN networked terminal sessions with non-blocking async I/O.
Enables unique session numbering, discovery, remote command execution, and cross-terminal signal linking.
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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
    connected_at: float
    last_heartbeat: float
    tags: List[str] = field(default_factory=list)


class IPCRouter:
    """High-Scale Multi-Client Async TCP Server handling up to 500+ concurrent terminal connections.
    Uses asyncio non-blocking event loop with TCP_NODELAY and socket reuse for ultra-low latency.
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
        # Next auto-increment session number
        self._next_session_id = 1
        self._session_lock = threading.Lock()

        # Shared variable store
        self.shared_memory: Dict[str, Any] = {}
        # Topic subscriptions: topic -> set of writers
        self.subscribers: Dict[str, Set[asyncio.StreamWriter]] = {}
        # Signal links: (source_session, source_var) -> List of (target_session, target_var)
        self.signal_links: Dict[str, List[Dict[str, Any]]] = {}

        # Pending remote execution promises: req_id -> asyncio.Future
        self._pending_exec_requests: Dict[str, asyncio.Future] = {}

    def start(self) -> bool:
        """Starts the router server in a dedicated background daemon thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="IPCRouterThread")
        self._thread.start()

        # Wait briefly for server socket to bind
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
                # Enable TCP_NODELAY to avoid 40ms Nagle buffering delay
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except Exception:
                pass

        peername = writer.get_extra_info("peername") or ("unknown", 0)
        session_id: Optional[int] = None

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
                    
                    info = TerminalSessionInfo(
                        session_id=session_id,
                        client_uuid=payload.get("client_uuid", str(uuid.uuid4())),
                        hostname=payload.get("hostname", socket.gethostname()),
                        ip_address=peername[0],
                        port=peername[1],
                        mode=payload.get("mode", "Circuit"),
                        active_file=payload.get("active_file", "untitled"),
                        connected_at=time.time(),
                        last_heartbeat=time.time(),
                        tags=payload.get("tags", [])
                    )
                    self.sessions[session_id] = info
                    self.client_session_map[writer] = session_id
                    self.session_writer_map[session_id] = writer

                    response["session_id"] = session_id
                    response["session_info"] = asdict(info)
                    response["total_active_sessions"] = len(self.sessions)

                # 2. List All Active Terminal Sessions on LAN
                elif msg_type == "list_sessions":
                    session_list = [asdict(s) for s in self.sessions.values()]
                    response["sessions"] = session_list
                    response["count"] = len(session_list)

                # 3. Remote Command Execution across Terminal Sessions
                elif msg_type == "remote_exec":
                    target_sid = payload.get("target_session_id")
                    command_str = payload.get("command", "")
                    target_writer = self.session_writer_map.get(target_sid)
                    
                    if not target_writer:
                        response["status"] = "error"
                        response["error"] = f"Session #{target_sid} is not connected to router."
                    else:
                        # Forward execution request to target session
                        exec_req_id = f"exec_{uuid.uuid4().hex[:8]}"
                        exec_msg = {
                            "type": "execute_command",
                            "id": exec_req_id,
                            "source_session_id": session_id,
                            "command": command_str
                        }
                        out_bytes = (json.dumps(exec_msg) + "\n").encode("utf-8")
                        target_writer.write(out_bytes)
                        await target_writer.drain()
                        response["status"] = "dispatched"
                        response["exec_req_id"] = exec_req_id

                # 4. Return Remote Command Execution Result
                elif msg_type == "exec_result":
                    # Broadcast or route result back to caller
                    origin_sid = payload.get("origin_session_id")
                    origin_writer = self.session_writer_map.get(origin_sid)
                    if origin_writer:
                        fwd = {"type": "remote_exec_complete", "payload": payload}
                        origin_writer.write((json.dumps(fwd) + "\n").encode("utf-8"))
                        await origin_writer.drain()

                # 5. Signal Linking between Terminal Sessions
                elif msg_type == "link_signal":
                    src_sig = payload.get("source_signal")
                    tgt_sid = payload.get("target_session_id")
                    tgt_sig = payload.get("target_signal")
                    link_key = f"{session_id}:{src_sig}"
                    if link_key not in self.signal_links:
                        self.signal_links[link_key] = []
                    self.signal_links[link_key].append({"target_session": tgt_sid, "target_signal": tgt_sig})
                    response["linked"] = f"Session #{session_id}:{src_sig} -> Session #{tgt_sid}:{tgt_sig}"

                # 6. Push Signal Value / Data Bridge
                elif msg_type == "push_signal":
                    sig_name = payload.get("signal_name")
                    val = payload.get("value")
                    link_key = f"{session_id}:{sig_name}"
                    # Forward to all linked targets
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

                # 7. Shared Variable Store
                elif msg_type == "set_var":
                    var_name = payload.get("name")
                    var_val = payload.get("value")
                    if var_name:
                        self.shared_memory[var_name] = var_val
                        await self._broadcast(f"var:{var_name}", {"name": var_name, "value": var_val})

                elif msg_type == "get_var":
                    response["value"] = self.shared_memory.get(payload.get("name"))

                elif msg_type == "broadcast":
                    b_msg = payload.get("message", "")
                    await self._broadcast("global", {"from_session": session_id, "message": b_msg})
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

    async def _broadcast(self, topic: str, payload: Any) -> None:
        """Sends broadcast to all connected terminal sessions."""
        msg = json.dumps({"type": "broadcast_event", "topic": topic, "payload": payload}) + "\n"
        out_bytes = msg.encode("utf-8")
        writers = list(self.session_writer_map.values())
        for w in writers:
            try:
                w.write(out_bytes)
            except Exception:
                pass


class IPCClient:
    """Client for terminal instance to connect to local or LAN IPCRouter.
    Maintains heartbeat, remote command execution handlers, and cross-terminal signal bridges.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.session_id: int = -1
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
        requested_session_id: Optional[int] = None
    ) -> bool:
        """Connects to IPC router synchronously with timeout."""
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.5)
            s.close()
        except Exception:
            return False

        self._thread = threading.Thread(
            target=self._run_client_loop,
            args=(mode, active_file, requested_session_id),
            daemon=True,
            name="IPCClientThread"
        )
        self._thread.start()

        # Wait for registration response
        for _ in range(30):
            if self.is_connected and self.session_id > 0:
                return True
            time.sleep(0.05)

        return self.is_connected

    def _run_client_loop(
        self,
        mode: str,
        active_file: str,
        requested_session_id: Optional[int]
    ) -> None:
        """Client asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._client_coro(mode, active_file, requested_session_id))

    async def _client_coro(
        self,
        mode: str,
        active_file: str,
        requested_session_id: Optional[int]
    ) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
            # Register session
            reg_msg = {
                "type": "register_session",
                "id": "reg_1",
                "payload": {
                    "requested_session_id": requested_session_id,
                    "client_uuid": self.client_uuid,
                    "hostname": socket.gethostname(),
                    "mode": mode,
                    "active_file": active_file
                }
            }
            self._writer.write((json.dumps(reg_msg) + "\n").encode("utf-8"))
            await self._writer.drain()

            self.is_connected = True

            # Listening loop
            while self.is_connected:
                line = await self._reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue

                # Handle registration response
                if msg.get("id") == "reg_1":
                    self.session_id = msg.get("session_id", 1)

                # Handle incoming remote execution command from another session
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

                    # Send result back
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

                # Handle signal update from linked session
                if msg.get("type") == "signal_update":
                    sig = msg.get("signal")
                    val = msg.get("value")
                    self.linked_signals[sig] = val

        except Exception:
            pass
        finally:
            self.is_connected = False

    def list_sessions_sync(self) -> List[Dict[str, Any]]:
        """Queries router for all active terminal sessions across the network."""
        if not self.is_connected:
            return []
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
            req = {"type": "list_sessions", "id": "list_req"}
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

    def link_signal_sync(self, local_signal: str, target_session_id: int, target_signal: str) -> Dict[str, Any]:
        """Links local signal to a remote session's signal input."""
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
                    "target_signal": target_signal
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

    def broadcast_sync(self, message: str) -> None:
        """Broadcasts text or variable change to all sessions."""
        if not self.is_connected:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            req = {
                "type": "broadcast",
                "id": "bcast_1",
                "payload": {"message": message}
            }
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            s.close()
        except Exception:
            pass


# Global singleton router and client
ipc_router_instance = IPCRouter()
ipc_client_instance = IPCClient()
