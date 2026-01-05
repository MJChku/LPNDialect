"""Shared helpers for protoacc workloads and token utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence


class CstStr(IntEnum):
  END_OF_MESSAGE = 0
  NONSCALAR = 1
  END_OF_MESSAGE_TOP_LEVEL = 2
  SCALAR = 3
  SUBMESSAGE = 4
  NONSUBMESSAGE = 5

  SCALAR_DISPATCH_REQ = 0
  STRING_GETPTR_REQ = 1
  STRING_LOADDATA_REQ = 2
  UNPACKED_REP_GETPTR_REQ = 3
  LOAD_NEW_SUBMESSAGE = 4
  LOAD_HASBITS_AND_IS_SUBMESSAGE = 5
  LOAD_EACH_FIELD = 6
  WRITE_OUT = 7

  READ = 0
  WRITE = 1


CONTROL_KEYS = ("type", "control_range", "repeated")
FIELD_META_KEYS = ("type", "num", "repeated")
UNIT_KEY = "bytes"


def meta_info_translate(name):
  if isinstance(name, CstStr):
    return name
  if isinstance(name, str):
    lowered = name.lower()
    if lowered == "nonscalar":
      return CstStr.NONSCALAR
    if lowered == "scalar":
      return CstStr.SCALAR
    if lowered == "submessage":
      return CstStr.SUBMESSAGE
  return CstStr(int(name))


def _align_bytes(value: int) -> int:
  if value % 16 == 0:
    return value
  return value + (16 - value % 16)


def _iterate_messages(payload) -> Iterable[Mapping]:
  if isinstance(payload, list):
    return payload
  return [payload]


def _parse_control_tokens(messages) -> List[MutableMapping[str, int]]:
  control_tokens: List[MutableMapping[str, int]] = []

  def flush(count_holder):
    count = count_holder[0]
    while count > 0:
      token_range = min(count, 32)
      control_tokens.append({
          "type": int(CstStr.NONSUBMESSAGE),
          "control_range": token_range,
          "repeated": 0,
      })
      count -= token_range
    count_holder[0] = 0

  def parse(node, depth: int) -> None:
    if not isinstance(node, Mapping):
      return
    sentinel = (CstStr.END_OF_MESSAGE_TOP_LEVEL
                if depth == 0 else CstStr.END_OF_MESSAGE)
    fields = [("f0", {
        "type": sentinel,
        "is_repeated": False,
        "data": [1],
    })]
    fields.extend(node.items())
    running = [0]
    for _, info in fields:
      field_type = meta_info_translate(info["type"])
      if field_type == CstStr.SUBMESSAGE:
        flush(running)
        control_tokens.append({
            "type": int(field_type),
            "control_range": 0,
            "repeated": 0,
        })
        subdata = info.get("data", {})
        if isinstance(subdata, list):
          for subentry in subdata:
            parse(subentry, depth + 1)
        elif isinstance(subdata, Mapping):
          parse(subdata, depth + 1)
      else:
        running[0] += 1
    flush(running)

  for message in _iterate_messages(messages):
    parse(message, depth=0)

  return control_tokens


def _parse_fields_meta(messages) -> List[MutableMapping[str, int]]:
  meta_tokens: List[MutableMapping[str, int]] = []

  def parse(node, depth: int) -> None:
    if not isinstance(node, Mapping):
      return
    sentinel = (CstStr.END_OF_MESSAGE_TOP_LEVEL
                if depth == 0 else CstStr.END_OF_MESSAGE)
    fields = [("f0", {
        "type": sentinel,
        "is_repeated": False,
        "data": [1],
    })]
    fields.extend(node.items())
    for _, info in fields:
      field_type = meta_info_translate(info["type"])
      is_repeated = 1 if info.get("is_repeated") else 0
      if field_type == CstStr.SUBMESSAGE:
        subdata = info.get("data", {})
        if isinstance(subdata, list):
          for subentry in subdata:
            parse(subentry, depth + 1)
        elif isinstance(subdata, Mapping):
          parse(subdata, depth + 1)
        continue
      values = info.get("data", [])
      if not isinstance(values, list):
        values = [values]
      meta_tokens.append({
          "type": int(field_type),
          "num": len(values),
          "repeated": is_repeated,
      })

  for message in _iterate_messages(messages):
    parse(message, depth=0)

  return meta_tokens


def _parse_units(messages) -> List[MutableMapping[str, int]]:
  unit_tokens: List[MutableMapping[str, int]] = []

  def parse(node, depth: int) -> None:
    if not isinstance(node, Mapping):
      return
    sentinel = (CstStr.END_OF_MESSAGE_TOP_LEVEL
                if depth == 0 else CstStr.END_OF_MESSAGE)
    fields = [("f0", {
        "type": sentinel,
        "is_repeated": False,
        "data": [1],
    })]
    fields.extend(node.items())
    for _, info in fields:
      field_type = meta_info_translate(info["type"])
      if field_type == CstStr.SUBMESSAGE:
        subdata = info.get("data", {})
        if isinstance(subdata, list):
          for subentry in subdata:
            parse(subentry, depth + 1)
        elif isinstance(subdata, Mapping):
          parse(subdata, depth + 1)
        continue
      values = info.get("data", [])
      if not isinstance(values, list):
        values = [values]
      for raw in values:
        try:
          qty = _align_bytes(int(raw))
        except (TypeError, ValueError):
          continue
        unit_tokens.append({"bytes": qty})

  for message in _iterate_messages(messages):
    parse(message, depth=0)

  return unit_tokens


def _parse_write_tokens(messages) -> List[MutableMapping[str, int]]:
  writes: List[MutableMapping[str, int]] = []

  def parse(node, depth: int) -> None:
    if not isinstance(node, Mapping):
      return
    sentinel = (CstStr.END_OF_MESSAGE_TOP_LEVEL
                if depth == 0 else CstStr.END_OF_MESSAGE)
    fields = [("f0", {
        "type": sentinel,
        "is_repeated": False,
        "data": [1],
    })]
    fields.extend(node.items())
    for _, info in fields:
      field_type = meta_info_translate(info["type"])
      if field_type == CstStr.SUBMESSAGE:
        subdata = info.get("data", {})
        if isinstance(subdata, list):
          for subentry in subdata:
            parse(subentry, depth + 1)
        elif isinstance(subdata, Mapping):
          parse(subdata, depth + 1)
        continue
      values = info.get("data", [])
      if not isinstance(values, list):
        values = [values]
      for raw in values:
        try:
          qty = _align_bytes(int(raw))
        except (TypeError, ValueError):
          continue
        writes.append({
            "level": depth,
            "bytes": qty,
        })

  for message in _iterate_messages(messages):
    parse(message, depth=0)

  return writes


@dataclass
class ProtoaccWorkload:
  control_tokens: List[MutableMapping[str, int]]
  field_meta_tokens: List[MutableMapping[str, int]]
  unit_tokens: List[MutableMapping[str, int]]
  writes: List[MutableMapping[str, int]]


def load_workload(path: Optional[Path | str]) -> ProtoaccWorkload:
  if path is None:
    raise ValueError("workload path is required")
  with open(Path(path), "r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return ProtoaccWorkload(control_tokens=_parse_control_tokens(payload),
                          field_meta_tokens=_parse_fields_meta(payload),
                          unit_tokens=_parse_units(payload),
                          writes=_parse_write_tokens(payload))
