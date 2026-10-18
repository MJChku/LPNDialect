"""Bind the timing prepass's decoded instructions to the canonical RTL image."""
import hashlib
import json
from pathlib import Path


def verified_program(directory,sid):
    directory=Path(directory)
    raw=(directory/f'program_{sid}.json').read_bytes()
    records=json.loads(raw)
    if not isinstance(records,list) or not records or any(
        not isinstance(row,list) or len(row)!=8 or any(type(w) is not int or not 0<=w<1<<32 for w in row)
        for row in records):
        raise ValueError('retained program must contain complete eight-word instructions')
    flattened=[w for row in records for w in row]
    # Canonical rtlsim serialization omits trailing zero words, including the
    # zero tail of HALT. Keep the one zero word for an all-zero image.
    while len(flattened)>1 and flattened[-1]==0:flattened.pop()
    canonical=[int(word,16) for word in (directory/f'prog_{sid}.hex').read_text().split()]
    if flattened!=canonical:
        raise ValueError('decoded timing program differs from canonical RTL instruction words')
    return records,hashlib.sha256(raw).hexdigest()
