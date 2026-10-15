"""Python-authored functional producer and a timing consumer, linked in C++."""

from lpnfunc import functional,Buffer,u32,i64
from lpnlang_mlir import NetBuilder,InputArray


@functional
def resolve_transfers(instructions:Buffer[u32],sizes:Buffer[i64],n:i64)->i64:
  for j in range(n):
    sizes[j] = i64(instructions[j]&65535)*4
  return n


def transfer_timing():
  net = NetBuilder("native_functional_bridge")
  sizes = InputArray("transfer_sizes",[4,32,64])
  cursor = net.place("cursor",initial_tokens=1)
  complete = net.place("program_completed",observable=True)

  @net.transition
  def transfer():
    guard(count(cursor)>0)
    idx = peek(cursor).get("id")
    guard(idx<index_cast(array_len(array(sizes)),src_type="index",dst_type="i64"))
    token = take(cursor)
    service = divi(array(sizes)[idx]+31,32)
    emit(cursor,token.set("id",idx+1),delay=sitofp(service+8))

  @net.transition
  def finish():
    guard(count(cursor)>0)
    guard(peek(cursor).get("id")==index_cast(array_len(array(sizes)),src_type="index",dst_type="i64"))
    take(cursor)
    emit(complete,create())

  return net
