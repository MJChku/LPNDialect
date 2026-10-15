"""Versioned immutable input tables shared by native producers and LPN models."""

import struct
import operator
from pathlib import Path


def write_input_tables(path,tables):
  """Write named i64 bit patterns; see include/LPN/Runtime/NativeInterface.h.

  This serializer also accepts NumPy integer arrays. No Python callback is
  involved while the timing simulation reads the resulting tables.
  """
  if len(tables)>10000:
    raise ValueError("too many input tables")
  path = Path(path)
  with path.open("wb") as out:
    out.write(b"LPNTBL1\0"+struct.pack("<Q",len(tables)))
    for name,values in tables.items():
      encoded = name.encode("utf-8")
      if not encoded or len(encoded)>1024 or b"\0" in encoded:
        raise ValueError("invalid input-table name")
      out.write(struct.pack("<QQ",len(encoded),len(values)))
      out.write(encoded)
      try:
        import numpy as np
      except ImportError:
        np = None
      if np is not None and isinstance(values,np.ndarray) and values.ndim==1 and values.dtype.kind in "iu" and values.dtype.itemsize==8:
        out.write(values.astype("<u8",copy=False).tobytes())
      else:
        out.write(struct.pack("<"+"Q"*len(values),*(operator.index(v)&0xffffffffffffffff for v in values)))
