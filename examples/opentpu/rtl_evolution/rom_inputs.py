"""Validate and fingerprint architecture-specific reciprocal initialization data."""
import hashlib
import json
from pathlib import Path


def reciprocal_inputs(table):
    table=Path(table).resolve()
    report=json.loads((table/'report.json').read_text())
    if (report['status']!='complete' or report['correction_bits']!=2 or report['inputs']!=1<<23
        or report['coarse_segments']!=4096 or not report['normalized_sequence_reconstructed_all_inputs']):
        raise ValueError('requires a complete two-bit reciprocal calibration')
    # These are the tables checked exhaustively against the original SV
    # normalized reciprocal. A self-consistent manifest alone is insufficient.
    verified={
        'coeff.hex':'2ed322e9987b727d73cb592a2b886b85c138095f4b7babfee27224477b3ee9a3',
        'correction.hex':'1d868c04b7226b703f299f2c3df4016fe3caf4be7b0c7af7a46a4fb7153fbf10'}
    hashes={}
    files=[table/'coeff.hex',table/'correction.hex']
    for f in files:
        hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        if hashes[f.name]!=verified[f.name] or hashes[f.name]!=report['files'][f.name]['sha256']:
            raise ValueError('reciprocal initialization data differs from its calibration')
    return files,hashes,[f'+recip_coeff={files[0]}',f'+recip_correction={files[1]}']
