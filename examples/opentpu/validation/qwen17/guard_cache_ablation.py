"""Compare guard caching on one immutable native OpenTPU timing input.

The full firing trace and final token marking must match. Performance runs
exclude tracing and alternate the order of the two modes. This measures native
timing only; it does not measure the numerical frontend or RTL execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    binary, inputs = args.binary.resolve(), args.inputs.resolve()
    fingerprints = {'binary_sha256': digest(binary), 'inputs_sha256': digest(inputs)}
    command = [str(binary), '--inputs=' + str(inputs), '--stats',
               '--wall-start=begin', '--wall-end=complete',
               '--max-time=1000000000000']
    records = []

    def run(cache, trace, iteration):
        mode = 'heartbeat' if cache else 'no_guard_cache'
        path = args.output / f'{mode}-{iteration}.log'
        actual = command + ([] if cache else ['--no-guard-cache'])
        if trace:
            actual += ['--trace', '--tokens']
        started = time.perf_counter()
        with path.open('w') as stream:
            result = subprocess.run(actual, stdout=stream, stderr=subprocess.STDOUT)
        process_seconds = time.perf_counter() - started
        result.check_returncode()
        log = path.read_text()
        wall = re.search(r'Wall window elapsed: ([0-9.]+) seconds', log)
        cycles = re.search(r'Simulation finished at t=([0-9.]+)', log)
        counts = re.search(r'Guard checks: (\d+); body executions: (\d+); rollbacks: (\d+)', log)
        hits = re.search(r'Guard cache hits: (\d+); dependency reads: (\d+)', log)
        if not all([wall, cycles, counts, hits]) or int(counts[3]) != 0:
            raise RuntimeError(f'Incomplete simulation or rollback: {path}')
        if 'place program_completed: 1 token(s)' not in log:
            raise RuntimeError(f'Missing program completion: {path}')
        semantic = [line for line in log.splitlines()
                    if line.startswith(('[t=', 'Simulation finished', '  place ', '    token '))]
        record = dict(mode=mode, trace=trace, iteration=iteration,
                      cycles=float(cycles[1]), core_seconds=float(wall[1]),
                      process_seconds=process_seconds, guard_checks=int(counts[1]),
                      body_executions=int(counts[2]), rollbacks=int(counts[3]),
                      guard_cache_hits=int(hits[1]), dependency_reads=int(hits[2]),
                      semantic_output_sha256=hashlib.sha256(('\n'.join(semantic)+'\n').encode()).hexdigest(),
                      firing_count=sum(line.startswith('[t=') for line in semantic),
                      log_sha256=digest(path))
        records.append(record)
        print(json.dumps(record), flush=True)
        return record

    reference = run(True, True, 'trace')
    control = run(False, True, 'trace')
    assert reference['firing_count'] > 0
    assert reference['semantic_output_sha256'] == control['semantic_output_sha256']
    assert reference['body_executions'] == control['body_executions']
    for iteration in range(args.repeat):
        for cache in ([True, False] if iteration % 2 == 0 else [False, True]):
            measured = run(cache, False, iteration)
            assert measured['cycles'] == reference['cycles']
            assert measured['body_executions'] == reference['body_executions']
    assert fingerprints == {'binary_sha256': digest(binary), 'inputs_sha256': digest(inputs)}
    medians = {mode: statistics.median(r['core_seconds'] for r in records
                                     if r['mode'] == mode and not r['trace'])
               for mode in ['heartbeat', 'no_guard_cache']}
    report = dict(status='complete', scope='Native timing only, copied Qwen command tables; no new RTL or numerical execution',
                  script_sha256=digest(Path(__file__)), **fingerprints,
                  firing_trace_and_final_marking_identical=True,
                  cycles=reference['cycles'], repeats=args.repeat, records=records,
                  median_core_seconds=medians,
                  native_speedup=medians['no_guard_cache']/medians['heartbeat'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
