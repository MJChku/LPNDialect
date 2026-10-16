"""Preserve and summarize live LPN/RTL observations without changing workers."""
from pathlib import Path
import csv,hashlib,json,os,shutil,time,traceback,datetime

def atomic_json(path,value):
    tmp=path.with_suffix('.monitor-tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:
        return False

def observation_row(case,record,failed=False):
    row={'case':case,'observation_status':'timing_failed' if failed else 'passed'}
    row.update({k:record.get(k) for k in [
        'position','phase','input_token','rtl_cycles','lpn_cycles','relative_cycle_error',
        'rtl_core_seconds','rtl_process_seconds','lpn_core_seconds','lpn_process_seconds',
        'functional_frontend_seconds','model_build_seconds','native_compile_seconds',
        'single_use_lpn_seconds','all_dram_and_tmem_bit_exact','identical_input_files',
        'zero_rollbacks','timing_cache_hit']})
    if row['position'] is None:row['position']=record.get('step')
    return row

def preserve_checkpoint(case,archive):
    source=Path(case['scratch_output'])
    state_path=source/'state.json'
    if not state_path.exists():return {'status':'no_checkpoint_yet'}
    state=json.loads(state_path.read_text())
    dest=archive/'recovery'/case['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'state.json').exists():
        old=json.loads((dest/'state.json').read_text())
        if old.get('snapshot_sha256')==state['snapshot_sha256']:
            return {'status':'current','position':state['position'],'directory':str(dest)}
    # One content-addressed snapshot plus a temporary new snapshot per active
    # case. Never replace the selected recovery manifest before verification.
    source_snapshot=source/state['snapshot']
    expected_size=source_snapshot.stat().st_size
    if shutil.disk_usage(dest).free<expected_size+(6<<30):
        return {'status':'insufficient_disk_for_new_checkpoint','position':state['position']}
    name='snapshot-'+state['snapshot_sha256']+'.npz'
    target=dest/name;temp=dest/(name+'.monitor-tmp')
    if not target.exists():
        shutil.copyfile(source_snapshot,temp)
        with temp.open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest()
            os.fsync(f.fileno())
        if actual!=state['snapshot_sha256']:
            temp.unlink()
            return {'status':'source_rotated_during_copy','position':state['position']}
        temp.replace(target)
    state['snapshot']=name
    atomic_json(dest/'state.json',state)
    report=source/'report.json'
    if report.exists():atomic_json(dest/'report.json',json.loads(report.read_text()))
    for previous in dest.glob('snapshot-*.npz'):
        if previous!=target:previous.unlink()
    return {'status':'verified','position':state['position'],'directory':str(dest)}

def main():
    root=Path.home()/'lpn-runs/qwen3-1.7b-parallel'
    summary_path=root/'overnight-summary.json'
    while True:
        try:
            batch=json.loads((root/'batch.json').read_text())
            rows=[];cases=[]
            for job in batch['jobs']:
                dest=root/'results'/job['id']
                report_path=dest/'report.json'
                case={k:job.get(k) for k in ['id','status','pid','completed_steps','generated_tokens']}
                case['worker_live']=bool(job.get('pid') and alive(job['pid']))
                if job['status']=='running' and case['worker_live']:
                    try:case['recovery']=preserve_checkpoint(job,root)
                    except (OSError,ValueError):case['recovery_error']=traceback.format_exc()
                elif job.get('archived_snapshot'):
                    # The supervisor checked this archive's content hash before
                    # recording it. Remove only this monitor's duplicate data.
                    case['recovery']={'status':'completed_archive','snapshot':job['archived_snapshot']}
                    recovery=root/'recovery'/job['id']
                    if recovery.exists():
                        for previous in recovery.glob('snapshot-*.npz'):previous.unlink()
                if report_path.exists():
                    report=json.loads(report_path.read_text())
                    for record in report['records']:
                        rows.append(observation_row(job['id'],record))
                failure=dest/'failed-step.json'
                if failure.exists():
                    case['failed_step']=json.loads(failure.read_text())
                    failed=observation_row(job['id'],case['failed_step'],failed=True)
                    if not any(r['case']==job['id'] and r['position']==failed['position'] for r in rows):
                        rows.append(failed)
                if job['status']=='failed':
                    case['last_log_lines']=(root/(job['id']+'.log')).read_text().splitlines()[-20:]
                cases.append(case)
            summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'monitor_pid':os.getpid(),'supervisor_pid':batch['supervisor_pid'],
                'supervisor_live':alive(batch['supervisor_pid']),
                'status':batch.get('status','running'),'cases':cases,
                'timing_observations':len(rows),
                'validated_token_steps':sum(r['observation_status']=='passed' for r in rows),
                'failed_timing_observations':sum(r['observation_status']=='timing_failed' for r in rows),
                'rtl_cycles':sum(r['rtl_cycles'] for r in rows),
                'rtl_core_seconds':sum(r['rtl_core_seconds'] for r in rows),
                'lpn_core_seconds':sum(r['lpn_core_seconds'] for r in rows),
                'full_lpn_seconds':sum(r['single_use_lpn_seconds'] for r in rows),
                'max_absolute_cycle_error':max((abs(r['relative_cycle_error']) for r in rows),default=None),
                'all_observed_steps_bit_exact':all(r['all_dram_and_tmem_bit_exact'] for r in rows) if rows else None,
                'all_observed_steps_within_5_percent':all(abs(r['relative_cycle_error'])<=.05 for r in rows) if rows else None,
                'disk_free_bytes':shutil.disk_usage(root).free,
                'scratch_free_bytes':shutil.disk_usage(batch['scratch_root']).free}
            if rows:
                summary['core_speedup']=summary['rtl_core_seconds']/summary['lpn_core_seconds']
                summary['full_speedup_vs_rtl_core']=summary['rtl_core_seconds']/summary['full_lpn_seconds']
                temp=root/'overnight-records.monitor-tmp'
                with temp.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                temp.replace(root/'overnight-records.csv')
            atomic_json(summary_path,summary)
            print(json.dumps({k:summary[k] for k in ['updated_utc','supervisor_live','validated_token_steps','rtl_core_seconds','max_absolute_cycle_error']}),flush=True)
            if summary['status'] in ('complete','finished_with_failures'):break
        except Exception:
            print(traceback.format_exc(),flush=True)
        time.sleep(30)

if __name__=='__main__':main()
