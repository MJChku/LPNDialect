"""Read-only audit of a completed RTL/LPN generation campaign and its archives."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--opentpu-root', type=Path, required=True)
    parser.add_argument('--expected-cases', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, project, opentpu = args.root.resolve(), args.project_root.resolve(), args.opentpu_root.resolve()
    batch, driver, summary = [read(root/name) for name in
                              ['batch.json', 'lpn-replays.json', 'overnight-summary.json']]
    require(batch.get('status') == 'complete', 'baseline batch is unfinished')
    require(len(batch['jobs']) == args.expected_cases, 'unexpected case count')
    require(all(j['status'] == 'complete' for j in batch['jobs']), 'a baseline case is unfinished or failed')
    require(driver['status'] == 'complete' and driver.get('all_replays_passed'), 'full replay batch is unfinished or failed')
    require(len(driver['cases']) == args.expected_cases, 'missing full generation replays')
    require(summary['status'] == 'complete' and summary['failed_timing_observations'] == 0, 'collector is unfinished or has failures')
    import numpy as np
    cases, originals, reference = {}, {}, None
    for job in batch['jobs']:
        name = job['id']
        require(name not in cases, f'duplicate case: {name}')
        dest = root/'results'/name
        baseline, state = read(dest/'report.json'), read(dest/'state.json')
        observation = driver['cases'][name]
        replay_dir = root/'lpn-replays'/name
        frozen, replay = read(replay_dir/'baseline.json'), read(replay_dir/'replay/report.json')
        identity, records = baseline['identity'], baseline['records']
        requested = job['requested_generated_tokens']
        require(baseline['status'] in ('complete', 'eos') and records, f'{name}: unfinished report')
        require(baseline['requested_generated_tokens'] == requested, f'{name}: requested output count differs')
        require(len(baseline['generated_tokens']) == requested or baseline['status'] == 'eos', f'{name}: output tokens missing')
        require(state['identity'] == identity and state['records'] == records, f'{name}: state/report differ')
        require(state['position'] == baseline['position'] == len(records), f'{name}: missing token steps')
        require([r['position'] for r in records] == list(range(len(records))), f'{name}: noncontiguous steps')
        require(state['input_tokens'] == [r['input_token'] for r in records], f'{name}: saved inputs differ')
        require(state['generated'] == baseline['generated_tokens'], f'{name}: generated state differs')
        prompt = identity['prompt_ids']
        require([r['input_token'] for r in records[:len(prompt)]] == prompt, f'{name}: prompt differs')
        require(all(r['phase'] == ('prefill' if i < len(prompt) else 'decode') for i, r in enumerate(records)), f'{name}: incorrect phases')
        require([r['input_token'] for r in records[len(prompt):]] == baseline['generated_tokens'], f'{name}: output sequence differs')
        for previous, current in zip(records, records[1:]):
            if current['phase'] == 'decode':
                require(current['input_token'] == previous['next_token'], f'{name}: decode is not greedy generation')
        if baseline['status'] == 'eos':
            config = read(Path(identity['model_directory'])/'config.json')
            generation = read(Path(identity['model_directory'])/'generation_config.json')
            allowed = []
            for value in [config.get('eos_token_id'), generation.get('eos_token_id')]:
                allowed.extend(value if isinstance(value, list) else [value])
            require(state.get('eos') in allowed and records[-1]['next_token'] == state['eos'], f'{name}: unverified EOS')
        for r in records:
            require(all(r[k] for k in ['all_dram_and_tmem_bit_exact', 'identical_input_files', 'accuracy_within_5_percent', 'zero_rollbacks']), f'{name}: failed numerical/timing observation')
            require(abs(r['lpn_cycles']/r['rtl_cycles']-1) <= .05, f'{name}: cycle error exceeds five percent')
            originals[name, r['position']] = r
        worker = process(job['pid'])
        require(worker is None or worker['state'] == 'Z' or worker['start_ticks'] != job['pid_start_ticks'], f'{name}: original worker is still live')
        snapshot = dest/state['snapshot']
        require(snapshot.parent == dest and digest(snapshot) == state['snapshot_sha256'], f'{name}: corrupt archive')
        with np.load(snapshot, allow_pickle=False) as saved:
            require(hashlib.sha256(saved['logits'].tobytes()).hexdigest() == records[-1]['logits_sha256'], f'{name}: archived logits differ')
        require(not (Path(batch['scratch_root'])/name).exists(), f'{name}: finished RAM bundle was not retired')
        require(observation['status'] == 'complete' and observation['returncode'] == 0, f'{name}: replay process failed')
        require(frozen == baseline and replay['baseline_identity'] == identity, f'{name}: replay reference differs')
        require(replay['baseline_report_sha256'] == digest(replay_dir/'baseline.json'), f'{name}: reference checksum differs')
        require(replay['complete_generation_replayed'] and replay['generated_tokens'] == baseline['generated_tokens'], f'{name}: incomplete generation replay')
        require(all(replay[k] for k in ['all_inputs_identical', 'all_logits_identical', 'all_predicted_cycles_identical', 'all_steps_within_5_percent']), f'{name}: replay differs')
        require(replay['steps'] == len(replay['records']) == len(records), f'{name}: replay steps missing')
        for original, measured in zip(records, replay['records']):
            require(all(original[k] == measured[k] for k in ['position', 'input_sha256', 'logits_sha256', 'lpn_cycles']), f'{name}: replay input/logit/cycle mismatch')
            require(measured['rollbacks'] == 0 and measured['guard_cache_hits'] > 0, f'{name}: heartbeat or rollback invariant failed')
        for total, key in [('rtl_cycles','rtl_cycles'), ('lpn_cycles','lpn_cycles'), ('rtl_core_seconds','rtl_core_seconds'), ('lpn_core_seconds','lpn_core_seconds')]:
            require(math.isclose(baseline[total], sum(r[key] for r in records), abs_tol=1e-8), f'{name}: baseline totals differ')
        require(replay['compiler_sha256'] == identity['lpn_compiler_sha256'] and replay['runtime_sha256'] == identity['lpn_runtime_sha256'], f'{name}: runtime identity changed')
        require(replay['replay_source_sha256'] == digest(project/'examples/opentpu/replay_e2e.py'), f'{name}: replay source changed')
        if reference is None:
            reference = identity
        for key in ['upstream_revision', 'checkpoint_sha256', 'config', 'lpn_source_sha256', 'lpn_compiler_sha256', 'lpn_runtime_sha256', 'lpn_frontend_sha256', 'native_inputs_sha256', 'rtl_source_sha256']:
            require(identity[key] == reference[key], f'{name}: campaign source/config differs: {key}')
        cases[name] = dict(steps=len(records), requested_generated_tokens=requested, generated_tokens=len(baseline['generated_tokens']), status=baseline['status'], snapshot_sha256=state['snapshot_sha256'], baseline_report_sha256=digest(dest/'report.json'), replay_report_sha256=digest(replay_dir/'replay/report.json'), all_inputs_logits_cycles_identical=True, guard_cache_observed=True, zero_rollbacks=True)
    for name, value in reference['checkpoint_sha256'].items():
        require(digest(Path(reference['model_directory'])/name) == value, f'checkpoint changed: {name}')
    for name, value in reference['lpn_source_sha256'].items():
        require(digest(project/'examples/opentpu'/name) == value, f'LPN model source changed: {name}')
    require(digest(project/'lpnlang_mlir/__init__.py') == reference['lpn_frontend_sha256'], 'frontend changed')
    require(digest(project/'lpnlang_mlir/native_inputs.py') == reference['native_inputs_sha256'], 'input binding source changed')
    for name, value in reference['rtl_source_sha256'].items():
        require(digest(opentpu/name) == value, f'RTL/observer source changed: {name}')
    require(subprocess.check_output(['git', '-C', str(opentpu), 'rev-parse', 'HEAD'], text=True).strip() == reference['upstream_revision'], 'upstream revision changed')
    command = batch['jobs'][0]['command']
    compiler = Path(command[command.index('--lpn-compiler')+1])
    require(digest(compiler) == reference['lpn_compiler_sha256'], 'compiler changed')
    require(digest(compiler.parent/'libLPNNativeRuntime.a') == reference['lpn_runtime_sha256'], 'runtime library changed')
    with (root/'overnight-records.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(row['case'], int(row['position'])) for row in rows]
    require(len(keys) == len(set(keys)) == len(originals) and set(keys) == set(originals), 'CSV observations missing or duplicated')
    for row, key in zip(rows, keys):
        require(row['observation_status'] == 'passed', 'CSV contains failed observations')
        original = originals[key]
        require(row['phase'] == original['phase'] and int(row['input_token']) == original['input_token'], 'CSV workload differs')
        for field in ['rtl_cycles','lpn_cycles','relative_cycle_error','rtl_core_seconds','lpn_core_seconds','single_use_lpn_seconds']:
            require(math.isclose(float(row[field]), original[field], abs_tol=1e-10), f'CSV timing differs: {field}')
    require(summary['validated_token_steps'] == len(originals) and summary['rtl_cycles'] == sum(r['rtl_cycles'] for r in originals.values()), 'collector totals differ')
    result = dict(status='complete', audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), audit_source_sha256=digest(Path(__file__)), model=batch['model'], checkpoint_revision=batch['checkpoint_revision'], cases=cases, token_steps=len(originals), generated_tokens=sum(c['generated_tokens'] for c in cases.values()), rtl_cycles=summary['rtl_cycles'], max_absolute_relative_cycle_error=max(abs(r['relative_cycle_error']) for r in originals.values()), all_archives_hash_verified=True, archived_logits_verified=True, all_model_compiler_runtime_rtl_fingerprints_verified=True, all_full_replays_verified=True, all_csv_observations_verified=True, csv_sha256=digest(root/'overnight-records.csv'))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
