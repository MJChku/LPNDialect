"""Pinned DSP48E2 binary-pin simulation model for integrated MXU gates."""
import hashlib
from pathlib import Path
import re
import subprocess

from .check_pair_pe import UNISIM_REVISION


def install(rtlsim, root):
    from examples.opentpu.e2e import digest
    root = Path(root).resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != UNISIM_REVISION:
        raise ValueError('unexpected vendor simulation-library revision')
    vendor = root/'verilog/src/unisims/DSP48E2.v'
    binary, count = re.subn(r"\(([A-Za-z0-9_]+(?:\[\d+\])?)\s+(===|!==)\s+1'bz\)",
        lambda m: "1'b0" if m[2] == '===' else "1'b1", vendor.read_text())
    if count != 168 or "1'bz" in binary:
        raise ValueError('unexpected vendor floating-input test structure')
    generated = rtlsim.BUILD/'lpn_vendor_binary'/('DSP48E2_'+hashlib.sha256(binary.encode()).hexdigest()+'.v')
    generated.parent.mkdir(parents=True, exist_ok=True)
    if generated.exists() and generated.read_text() != binary:
        raise ValueError('generated vendor model differs')
    if not generated.exists(): generated.write_text(binary)
    # The primitive resolves glbl.GSR hierarchically. Instantiate the official
    # global module in a simulation-only harness copy; do not edit frozen RTL.
    original_tb = rtlsim.TB
    harness = original_tb/'tb_top.sv'
    text = harness.read_text()
    if text.count('module tb_top;') != 1:
        raise ValueError('unexpected top harness declaration')
    overlay_text = text.replace('module tb_top;', 'module tb_top;\n  glbl glbl();')
    identity = hashlib.sha256((overlay_text+digest(original_tb/'lpn_wall.cpp')).encode()).hexdigest()
    overlay = rtlsim.BUILD/'lpn_vendor_harness'/identity
    overlay.mkdir(parents=True, exist_ok=True)
    for name, content in [('tb_top.sv',overlay_text),('lpn_wall.cpp',(original_tb/'lpn_wall.cpp').read_text())]:
        path=overlay/name
        if path.exists() and path.read_text()!=content: raise ValueError('vendor harness changed')
        if not path.exists(): path.write_text(content)
    rtlsim.TB=overlay
    sources = [root/'verilog/src/glbl.v', generated]
    for path in sources:
        if str(path) not in rtlsim.RTL_SOURCES: rtlsim.RTL_SOURCES.append(str(path))
    return dict(root=str(root), revision=revision,
        binary_driven_pins_only=True, floating_input_tests_replaced=count,
        arithmetic_and_register_logic_unchanged=True,
        harness_scope='Original top plus official global-reset instance only',
        source_sha256={str(p): digest(p) for p in [vendor, *sources, harness,
            original_tb/'lpn_wall.cpp',overlay/'tb_top.sv',overlay/'lpn_wall.cpp',Path(__file__)]})
