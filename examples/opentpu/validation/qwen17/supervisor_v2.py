"""Adopt the existing Qwen workers and safely expand the overnight queue."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time,traceback

def atomic(path,value):
    temp=path.with_suffix('.supervisor-tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def proc(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':fields[0],'start_ticks':int(fields[19])}
    except FileNotFoundError:
        return None

def live(job):
    p=proc(job['pid'])
    return p is not None and p['state']!='Z' and p['start_ticks']==job['pid_start_ticks']

def successful_report(out,job):
    try:
        r=json.loads((out/'report.json').read_text())
        records=r['records']
        return (r['status'] in ('complete','eos') and records and r['position']==len(records)
            and [x['position'] for x in records]==list(range(len(records)))
            and (r['status']=='eos' or len(r['generated_tokens'])>=job['requested_generated_tokens'])
            and r['all_steps_within_5_percent']
            and all(x['all_dram_and_tmem_bit_exact'] and x['identical_input_files'] and x['zero_rollbacks'] for x in records))
    except (OSError,ValueError,KeyError):
        return False

def archive_case(out,dest,recovery):
    state=json.loads((out/'state.json').read_text())
    selected=out/state['snapshot'];target=dest/state['snapshot']
    # Prefer a hard link to the monitor's verified immutable content, so the
    # final archive does not duplicate gigabytes of disk recovery data.
    recovered=None
    if (recovery/'state.json').exists():
        old=json.loads((recovery/'state.json').read_text())
        if old['snapshot_sha256']==state['snapshot_sha256']:
            candidate=recovery/old['snapshot']
            if candidate.exists():recovered=candidate
    if recovered is None and shutil.disk_usage(dest).free<selected.stat().st_size+(6<<30):
        return False
    temp=dest/(target.name+'.final-archive-tmp')
    if temp.exists():temp.unlink()
    try:
        if recovered is not None:os.link(recovered,temp)
        else:shutil.copyfile(selected,temp)
    except FileNotFoundError:
        # A monitor can rotate its old recovery file between lookup and link.
        return False
    with temp.open('rb') as f:
        actual=hashlib.file_digest(f,'sha256').hexdigest();os.fsync(f.fileno())
    if actual!=state['snapshot_sha256']:raise RuntimeError('Final archive hash differs')
    temp.replace(target)
    atomic(dest/'state.json',state)
    atomic(dest/'report.json',json.loads((out/'report.json').read_text()))
    shutil.rmtree(out)
    return str(target)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--handoff',type=Path,required=True)
    parser.add_argument('--parallel',type=int,default=6)
    a=parser.parse_args()
    home=Path.home();archive=home/'lpn-runs/qwen3-1.7b-parallel'
    scratch=Path('/dev/shm/lpn-qwen3-17-validation');project=home/'LPNDialect-qwen-validation'
    lock=(archive/'supervisor-v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=json.loads((archive/'batch.json').read_text())
    handoff=json.loads(a.handoff.read_text())
    assert original['supervisor_pid']==handoff['old_supervisor_pid']
    handles={};adopted=[]
    for job in original['jobs']:
        if job['status']=='running':
            current=proc(job['pid'])
            if current is not None:
                job['pid_start_ticks']=current['start_ticks']
                adopted.append({'id':job['id'],'pid':job['pid'],'start_ticks':current['start_ticks']})
            else:
                job['pid_start_ticks']=handoff['workers'][job['id']]['start_ticks']
            job['adopted_worker']=True
    ready=a.handoff.with_suffix('.ready.json')
    atomic(ready,{'new_supervisor_pid':os.getpid(),'adopted':adopted,'parallel':a.parallel})
    commit=a.handoff.with_suffix('.commit')
    while not commit.exists():time.sleep(.1)
    original['supervisor_pid']=os.getpid();original['parallel_cases']=a.parallel
    original.setdefault('parallelism_history',[]).append({'unix':time.time(),'parallel':a.parallel,
        'reason':'Measured headroom after twenty passing comparisons; existing workers adopted without restart',
        'adopted_worker_pids':[j['pid'] for j in original['jobs'] if j.get('adopted_worker')]})
    env={**os.environ,'PYTHONPATH':str(project),'PYTHONDONTWRITEBYTECODE':'1',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','OTPU_BUILD_JOBS':'4',
        'OTPU_IMG_CACHE':str(scratch/'image-cache'),'OTPU_IMG_CACHE_GB':'4','OTPU_SIM_TMP':str(scratch/'tmp'),
        'VERILATOR_ROOT':str(home/'.local/opt/lpn-verilator-5.052'),
        'PATH':str(home/'.local/opt/lpn-verilator-5.052/bin')+':'+os.environ['PATH']}
    atomic(archive/'batch.json',original)
    while any(j['status'] in ('queued','running','archiving') for j in original['jobs']):
        admitted=0
        active=sum(j['status'] in ('running','archiving') for j in original['jobs'])
        for job in original['jobs']:
            if job['status']!='queued' or active>=a.parallel:continue
            memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            # Account for workers admitted in this iteration before they have
            # allocated their peak memory, rather than overcommitting in a burst.
            if memory<(52+20*admitted)*(1<<30) or shutil.disk_usage(scratch).free<(20+8*admitted)*(1<<30):break
            out=scratch/job['id'];out.mkdir(exist_ok=True)
            command=[str(home/'FeSens-openTPU/.venv/bin/python'),'-m','examples.opentpu.e2e',
                '--opentpu-root',str(home/'FeSens-openTPU-qwen'),'--model-dir',str(home/'FeSens-openTPU/models/Qwen3-1.7B'),
                '--lpn-compiler',str(home/'LPNDialect-heartbeat/build/tools/lpn-compile/lpn-compile'),
                '--output',str(out),'--depth','128','--cap','128','--prompt',job['prompt'],
                '--tokens',str(job['requested_generated_tokens']),'--reuse-timing','--command-capacity','32768','--rtl-timeout','28800']
            if (out/'state.json').exists():command.append('--resume')
            with (archive/(job['id']+'.log')).open('a') as log:
                child=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            handles[job['id']]=child
            job.update(status='running',pid=child.pid,pid_start_ticks=proc(child.pid)['start_ticks'],
                command=command,started_unix=time.time(),scratch_output=str(out))
            print(json.dumps({'started':job['id'],'pid':child.pid}),flush=True)
            active+=1;admitted+=1
        for job in original['jobs']:
            if job['status'] not in ('running','archiving'):continue
            out=Path(job['scratch_output']);dest=archive/'results'/job['id'];dest.mkdir(parents=True,exist_ok=True)
            for name in ('report.json','state.json','failed-step.json'):
                if (out/name).exists():
                    value=json.loads((out/name).read_text());atomic(dest/name,value)
                    if name=='report.json':job.update(completed_steps=value['position'],
                        generated_tokens=len(value['generated_tokens']),rtl_core_seconds=value['rtl_core_seconds'],
                        all_steps_within_5_percent=value['all_steps_within_5_percent'])
            if job['status']=='running':
                if job.get('adopted_worker'):
                    if live(job):continue
                    job['returncode']=None
                    job['exit_status_source']='Validated terminal report after adopted worker exited; original exit code unavailable'
                else:
                    code=handles[job['id']].poll()
                    if code is None:continue
                    job['returncode']=code
                    job['exit_status_source']='Popen return code'
                job['worker_ended_unix']=time.time()
                success=successful_report(out,job) and job.get('returncode') in (None,0)
                job['status']='archiving' if success else 'failed'
            if job['status']=='archiving':
                try:
                    path=archive_case(out,dest,archive/'recovery'/job['id'])
                    if path:job.update(status='complete',archived_snapshot=path,ended_unix=time.time())
                except Exception:
                    job['archive_error']=traceback.format_exc()
            if job['status'] in ('complete','failed'):
                print(json.dumps({'finished':job['id'],'status':job['status']}),flush=True)
        original['updated_unix']=time.time();atomic(archive/'batch.json',original)
        if any(j['status'] in ('queued','running','archiving') for j in original['jobs']):time.sleep(15)
    original['status']='complete' if all(j['status']=='complete' for j in original['jobs']) else 'finished_with_failures'
    atomic(archive/'batch.json',original)

if __name__=='__main__':main()
