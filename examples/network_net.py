"""Network-style LPN example built with the MLIR DSL."""

from __future__ import annotations

import argparse
from typing import Iterable, List

from lpnlang_mlir import NetBuilder, PlaceHandle


LINK_DELAY_NS = 130.0
LINK_BW_GBPS = 100.0
SWITCH_DELAY_NS = 100.0
NUM_DEVS = 64
PAYLOAD_BYTES = 8_000_000


class NetUniDirLink:
  """Point-to-point link with explicit serialization delay."""

  def __init__(self,
               net: NetBuilder,
               name: str,
               *,
               delay_ns: float = LINK_DELAY_NS,
               bandwidth_gbps: float = LINK_BW_GBPS):
    self._net = net
    self.name = name
    self.delay_ns = delay_ns
    self.bandwidth = bandwidth_gbps
    self.ctrl = net.place(f"{name}_ctrl", initial_tokens=1)
    self._connected = False

  def connect(self, src: PlaceHandle, dst: PlaceHandle) -> None:
    if self._connected:
      raise ValueError(f"{self.name} already wired")
    transition_name = f"{self.name}_transfer"

    @self._net.transition(transition_name)
    def forward(src=src, dst=dst):
      ctrl_token = take(self.ctrl)
      payload = take(src)
      size_i64 = payload.get("size")
      serialization = sitofp(size_i64) / self.bandwidth
      total_delay = serialization + self.delay_ns
      emit(dst, payload, delay=total_delay)
      emit(self.ctrl, ctrl_token, delay=total_delay)

    self._connected = True


class NetDevice:
  def __init__(self, net: NetBuilder, dev_id: int, seed_pool: PlaceHandle):
    self._net = net
    self.id = f"d{dev_id}"
    self.server_id = dev_id
    self.out_buf = net.place(f"{self.id}_out", observable=True)
    self.in_buf = net.place(f"{self.id}_in", observable=True)
    self.link_out = NetUniDirLink(net, f"{self.id}_tx")
    self.link_in = NetUniDirLink(net, f"{self.id}_rx")
    self._workload_idx = 0
    self._seed_pool = seed_pool

  def attach(self, tor: "NetToRSwitch", port_id: int) -> None:
    tor_ingress = tor.ingress(port_id)
    tor_egress = tor.egress(port_id)
    self.link_out.connect(self.out_buf, tor_ingress)
    self.link_in.connect(tor_egress, self.in_buf)

  def inject_workload(self, dst: int, size: int, count: int = 1) -> None:
    latch = self._net.place(
        f"{self.id}_inject_ctrl_{self._workload_idx}", initial_tokens=1)
    transition_name = f"{self.id}_inject_{self._workload_idx}"
    workload_idx = self._workload_idx
    self._workload_idx += 1

    @self._net.transition(transition_name)
    def inject(latch=latch,
               dst=dst,
               size=size,
               count=count,
               workload_idx=workload_idx):
      take(latch)
      for seq in range(count):
        seed = take(self._seed_pool)
        token = seed.set("src", self.server_id)
        token = token.set("dst", dst)
        token = token.set("size", size)
        seq_i64 = index_cast(seq, src_type="index", dst_type="i64")
        token_id = seq_i64 + workload_idx * 1_000
        token = token.set("id", token_id)
        emit(self.out_buf, token)


class NetToRSwitch:
  """Simplified top-of-rack switch with per-port schedulers."""

  def __init__(self,
               net: NetBuilder,
               name: str,
               port_ids: Iterable[int],
               *,
               delay_ns: float = SWITCH_DELAY_NS,
               server_start: int = 0):
    self._net = net
    self.id = name
    self.delay_ns = delay_ns
    self.server_start = server_start
    self.ctrl = net.place(f"{name}_ctrl", initial_tokens=1)
    self.port_ids = list(port_ids)
    self._port_to_index = {}
    for idx in range(len(self.port_ids)):
      pid = self.port_ids[idx]
      self._port_to_index[pid] = idx
    self._ingress: List[PlaceHandle] = []
    self._egress: List[PlaceHandle] = []
    for pid in self.port_ids:
      self._ingress.append(net.place(f"{name}_ingress_{pid}"))
      self._egress.append(net.place(f"{name}_egress_{pid}"))
    for slot in range(len(self.port_ids)):
      pid = self.port_ids[slot]
      self._build_forwarding(slot, pid)

  def ingress(self, port_id: int) -> PlaceHandle:
    return self._ingress[self._port_to_index[port_id]]

  def egress(self, port_id: int) -> PlaceHandle:
    return self._egress[self._port_to_index[port_id]]

  def _build_forwarding(self, slot: int, port_id: int) -> None:
    transition_name = f"{self.id}_forward_port_{port_id}"

    @self._net.transition(transition_name)
    def forward(slot=slot):
      ctrl = take(self.ctrl)
      packet = take(self._ingress[slot])
      dst = packet.get("dst")
      relative = dst - self.server_start
      idx = index_cast(relative, src_type="i64")
      egress_handles = array(self._egress)
      target = egress_handles[idx]
      packet = packet.set("hops", 1)
      delay = self.delay_ns
      emit(target, packet, delay=delay)
      emit(self.ctrl, ctrl, delay=delay)


def create_network_net(num_devices: int = 2,
                       messages_per_device: int = 1) -> NetBuilder:
  net = NetBuilder("network")
  total_tokens = num_devices * messages_per_device
  seed_pool = net.place("token_seed", initial_tokens=total_tokens)
  tor = NetToRSwitch(net, "tor", range(num_devices), server_start=0)
  devices: List[NetDevice] = []
  for dev_id in range(num_devices):
    device = NetDevice(net, dev_id, seed_pool)
    device.attach(tor, dev_id)
    devices.append(device)

  for dev_idx in range(len(devices)):
    device = devices[dev_idx]
    dst = (dev_idx + 1) % num_devices
    device.inject_workload(dst=dst,
                           size=PAYLOAD_BYTES,
                           count=messages_per_device)
  return net


def build_network_example(num_devices: int = 2,
                          messages_per_device: int = 1) -> str:
  return create_network_net(num_devices, messages_per_device).build()


def simulate_network(num_devices: int,
                     messages_per_device: int,
                     *,
                     max_time: float,
                     debug: bool = False) -> float:
  net = create_network_net(num_devices, messages_per_device)
  sim = net.python_simulator()
  final_time = sim.run(max_time=max_time, debug=debug)
  print(f"Simulation finished at t={final_time:.3f} ns")
  observables = [handle for handle, _cap, _init, observable in net._places
                 if observable]
  if not observables:
    observables = [handle for handle, *_rest in net._places]
  for handle in observables:
    tokens = sim.place_dicts(handle)
    print(f"  place {handle.name}: {len(tokens)} token(s)")
    for token in tokens:
      print(f"    token {token}")
  return final_time


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--devices",
                      type=int,
                      default=2,
                      help="Number of devices in the fabric")
  parser.add_argument("--messages",
                      type=int,
                      default=1,
                      help="Messages per device")
  parser.add_argument("--simulate",
                      action="store_true",
                      help="Run the Python simulator instead of emitting MLIR")
  parser.add_argument("--max-time",
                      type=float,
                      default=10000000.0,
                      help="Simulation time budget (ns)")
  parser.add_argument("--debug",
                      action="store_true",
                      help="Print every transition firing while simulating")
  args = parser.parse_args()
  if args.simulate:
    simulate_network(args.devices,
                     args.messages,
                     max_time=args.max_time,
                     debug=args.debug)
  else:
    print(build_network_example(args.devices, args.messages))
