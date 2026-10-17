"""Audit the full original/evolved 128/32 RTL comparison and timing replay.

This consumes completed experiments. It does not turn preparation records or
primitive gates into checkpoint validation evidence.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equal_files(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while block := a.read(8 << 20):
            if block != b.read(len(block)):
                return False
    return True


def audit(baseline, evolved, replay, baseline_root, evolved_root):
    original = json.loads((baseline / 'report.json').read_text())
    candidate = json.loads((evolved / 'report.json').read_text())
    corrected = json.loads((replay / 'report.json').read_text())
    require(original['status'] == candidate['status'] == corrected['status'] == 'complete',
            'all three full experiments must be complete')
    require(original['mode'] == 'validate-rtl', 'baseline must contain actual RTL observations')
    require(original['prompt_tokens'] == candidate['prompt_tokens'] == 128,
            'expected 128 prompt tokens')
    require(original['returned_output_tokens'] == candidate['output_tokens'] == 32,
            'expected 32 actual generated tokens')
    require(len(original['prompt_ids']) == 128 and len(original['generated_ids']) == 32,
            'token counts disagree with actual IDs')
    require(original['prompt_ids'] == candidate['prompt_ids'] and
            original['generated_ids'] == candidate['generated_ids'], 'inference IDs differ')
    require(original['final_position'] == candidate['final_position'] == 160,
            'upstream inference did not advance through the complete workload')
    require(original['config'] == candidate['config'], 'software configurations differ')
    require(Path(corrected['case']).resolve() == baseline.resolve(), 'timing replay targets another case')
    require(original['entrypoint'] == candidate['entrypoint'] == 'unmodified upstream Engine.generate',
            'unexpected inference entrypoint')
    require(candidate['baseline_report_sha256'] == digest(baseline / 'report.json'),
            'original report changed after comparison')
    for case, root in [(original, baseline_root), (candidate, evolved_root)]:
        for name, expected in case['rtl_source_sha256'].items():
            require(digest(root / name) == expected, f'RTL source changed: {root / name}')
    for name, expected in corrected['source_sha256'].items():
        require(digest(Path(name)) == expected, f'timing source changed: {name}')
    originals = original['kernels']
    candidates = candidate['kernels']
    timings = {r['kernel_index']: r for r in corrected['records']}
    require(len(originals) == len(candidates) == len(timings) == 48,
            'expected all 16 batched prefill and 32 decode kernels')
    config = original['config']
    identity = json.loads((baseline / 'identity.json').read_text())
    baseline_binaries = {str(p): digest(p) for p in
                         (baseline_root / 'build/verilator').glob('tb_top_*/Vtb_top')}
    require(identity['binary_sha256'] in baseline_binaries.values(),
            'retained original simulator binary is missing or changed')
    phases = {phase: dict(baseline_cycles=0, evolved_cycles=0,
                          original_lpn_cycles=0, evolved_lpn_cycles=0, kernels=0)
              for phase in ('prefill', 'decode')}
    binaries = {}
    observations = []
    for index, (old, new) in enumerate(zip(originals, candidates)):
        directory = evolved / 'kernels' / f'{index:04d}'
        worker = json.loads((directory / 'rtl-result.json').read_text())
        saved = json.loads((directory / 'record.json').read_text())
        timing = timings[index]
        phase = 'prefill' if index < 16 else 'decode'
        position = index * 8 if index < 16 else 128 + index - 16
        require(old['step'] == new['step'] == worker['step'] == index,
                f'kernel {index}: indices differ')
        retained = json.loads((baseline / 'kernels' / f'{index:04d}' / 'record.json').read_text())
        require(retained == old and old['baseline_binary_sha256'] == identity['binary_sha256'],
                f'kernel {index}: retained original evidence differs')
        for record in (old, new, saved):
            require(record['position_before'] == position and record['phase'] == phase and
                    record['input_rows'] == (8 if phase == 'prefill' else 1),
                    f'kernel {index}: upstream schedule differs')
            require(record['all_dram_and_tmem_bit_exact'] and record['identical_input_files'] and
                    record['zero_rollbacks'], f'kernel {index}: verification is missing')
        require(new['rtl_verification_pending'] is False, f'kernel {index}: still pending')
        require(old['input_sha256'] == new['input_sha256'] == worker['input_sha256'] and
                old['args'] == new['args'] and old['instructions'] == new['instructions'],
                f'kernel {index}: original and evolved workload inputs differ')
        require(timing['rtl_cycles'] == old['rtl_cycles'] and timing['program_hash_verified'] and
                timing['zero_rollbacks'], f'kernel {index}: corrected baseline timing evidence differs')
        for key in ('rtl_cycles', 'lpn_cycles', 'baseline_rtl_cycles', 'simulator'):
            require(new[key] == worker[key] == saved[key], f'kernel {index}: {key} differs')
        require(worker['baseline_rtl_cycles'] == old['rtl_cycles'], 'baseline cycle reference differs')
        for name, expected in new['input_sha256'].items():
            require(digest(directory / name) == expected, f'kernel {index}: input changed: {name}')
        for sid in range(config['S']):
            require(equal_files(directory / f'dram_out_{sid}.bin', directory / f'golden_dram_{sid}.bin'),
                    f'kernel {index}: full DRAM comparison failed')
            words = array('I', (int(x, 16) for x in
                              (directory / f'tmem_{sid}.hex').read_text().split()))
            require(words.itemsize == 4 and len(words) == config['TMEM_WORDS'],
                    f'kernel {index}: incomplete scratchpad dump')
            if sys.byteorder != 'little':
                words.byteswap()
            require(words.tobytes() == (directory / f'golden_tmem_{sid}.bin').read_bytes(),
                    f'kernel {index}: full scratchpad comparison failed')
        log = (directory / 'rtl.log').read_text()
        result = re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)', log)
        require(result is not None and tuple(map(int, result.groups())) == (new['rtl_cycles'], 1, 0),
                f'kernel {index}: raw RTL completion/cycles differ')
        counts = [int(v) for v in re.findall(r'SLICE \d+ icount=(\d+)', log)]
        require(counts == new['instructions'], f'kernel {index}: raw RTL instruction count differs')
        for path, expected in [(directory / 'lpn.log', new['lpn_cycles']),
                               (replay / f'{index:04d}' / 'lpn.log', timing['new_lpn_cycles'])]:
            native = path.read_text()
            cycles = re.search(r'Simulation finished at t=([0-9.]+)', native)
            require(cycles is not None and float(cycles[1]) == expected and 'rollbacks: 0' in native and
                    'place program_completed: 1 token(s)' in native,
                    f'kernel {index}: native completion evidence differs: {path}')
        executable = Path(new['simulator'])
        if str(executable) not in binaries:
            binaries[str(executable)] = digest(executable)
        require(binaries[str(executable)] == worker['baseline_binary_sha256'],
                f'kernel {index}: evolved simulator binary differs')
        values = phases[phase]
        for name, value in [('baseline_cycles', old['rtl_cycles']), ('evolved_cycles', new['rtl_cycles']),
                            ('original_lpn_cycles', timing['new_lpn_cycles']),
                            ('evolved_lpn_cycles', new['lpn_cycles'])]:
            values[name] += value
        values['kernels'] += 1
        observations.append(dict(kernel=index, phase=phase, baseline_cycles=old['rtl_cycles'],
            evolved_cycles=new['rtl_cycles'], original_lpn_error=timing['relative_cycle_error'],
            evolved_lpn_error=new['lpn_cycles'] / new['rtl_cycles'] - 1))
    total = {name: sum(p[name] for p in phases.values()) for name in next(iter(phases.values()))}
    require(total['baseline_cycles'] == original['rtl_cycles'] == candidate['baseline_rtl_cycles'] and
            total['evolved_cycles'] == candidate['evolved_rtl_cycles'], 'full cycle sums differ')
    require(total['evolved_lpn_cycles'] == candidate['lpn_cycles'], 'evolved LPN cycle sum differs')
    for values in [*phases.values(), total]:
        values['architecture_cycle_speedup'] = values['baseline_cycles'] / values['evolved_cycles']
        values['original_lpn_relative_error'] = values['original_lpn_cycles'] / values['baseline_cycles'] - 1
        values['evolved_lpn_relative_error'] = values['evolved_lpn_cycles'] / values['evolved_cycles'] - 1
    return dict(status='complete', scope=__doc__, prompt_tokens=128, output_tokens=32,
        numerical_validation='All evolved raw DRAM/TMEM dumps freshly compared to independent ISA; '
                             'all original retained per-kernel numerical/input evidence required.',
        all_inputs_identical=True, generated_ids=original['generated_ids'], final_position=160,
        phases=phases, total=total, observations=observations,
        simulator_sha256=dict(original=baseline_binaries, evolved=binaries),
        worst_original_lpn_error=max(abs(r['original_lpn_error']) for r in observations),
        worst_evolved_lpn_error=max(abs(r['evolved_lpn_error']) for r in observations),
        report_sha256={str(p / 'report.json'): digest(p / 'report.json')
                       for p in (baseline, evolved, replay)},
        interpretation='Hardware cycles at a common clock; FPGA resource fit, Fmax and a vendor '
                       'HBM controller are not established. Host validation processes run independently; '
                       'the accelerator workload latency is the sum of all 48 kernels.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'evolved', 'replay', 'baseline-root', 'evolved-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline, args.evolved, args.replay, args.baseline_root, args.evolved_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['total'], indent=2))


if __name__ == '__main__':
    main()
