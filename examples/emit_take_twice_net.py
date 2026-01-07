"""Tiny demo: emit once to B, then take twice from B, then emit twice to C.

B starts with one initial token; a bootstrap transition emits a second token to B.
A consumer transition takes two tokens from B and forwards both to C.
"""

from __future__ import annotations

from lpnlang_mlir import NetBuilder

# Placeholders so linters understand injected names inside @net.transition bodies.
create = None  # type: ignore
emit = None    # type: ignore
take = None    # type: ignore


def build_emit_take_twice_net() -> str:
  net = NetBuilder("emit_take_twice")

  # Places
  seed = net.place("seed", initial_tokens=1, observable=True)
  b = net.place("B")
  c = net.place("C", observable=True)

  @net.transition
  def seed_b():
    """Runs once to add a single token into B."""
    t = take(seed)
    emit(b, t)

  @net.transition
  def consume_b_twice():
    """Takes two tokens from B and emits both to C."""
    first = take(b)
    second = take(b)
    emit(c, first.set("src", 1))
    emit(c, second.set("src", 2))

  return net.build()


if __name__ == "__main__":
  print(build_emit_take_twice_net())
