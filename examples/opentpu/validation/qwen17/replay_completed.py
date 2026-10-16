"""Measure full LPN-only wall time once each supervised Qwen case finishes."""
from pathlib import Path
import json,os,shutil,subprocess,time,traceback

def atomic(path,value):
    tmp=path.with_suffix('.replay-tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    home=Path.home();root=home/'lpn-runs/qwen3-1.7b-parallel'
    project=home/'LPNDialect-qwen-validation';output=root/'lpn-replays';output.mkdir(exist_ok=True)
    state={'driver_pid':os.getpid(),'status':'waiting_for_complete_cases','cases':{}}
    state_path=root/'lpn-replays.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous.get('status')!='complete':
            raise RuntimeError('Existing replay driver state must be inspected before another driver starts')
        state['cases']=previous['cases']
    env={**os.environ,'PYTHONPATH':str(project),'PYTHONDONTWRITEBYTECODE':'1',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','OTPU_BUILD_JOBS':'4',
        'OTPU_IMG_CACHE':'/dev/shm/lpn-qwen3-17-validation/image-cache'}
    while True:
        batch=json.loads((root/'batch.json').read_text())
        for case in batch['jobs']:
            name=case['id']
            if name in state['cases']:continue
            if case['status']=='failed':
                state['cases'][name]={'status':'skipped_baseline_failed'};continue
            if case['status']!='complete':continue
            baseline=root/'results'/name/'report.json'
            report=json.loads(baseline.read_text())
            if report['status'] not in ('complete','eos'):
                state['cases'][name]={'status':'failed','error':'Completed job has an unfinished generation report'};continue
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            if available<32*(1<<30) or shutil.disk_usage(output).free<8*(1<<30):continue
            dest=output/name;dest.mkdir(exist_ok=True)
            frozen=dest/'baseline.json';frozen.write_bytes(baseline.read_bytes())
            command=[str(home/'FeSens-openTPU/.venv/bin/python'),'-m','examples.opentpu.replay_e2e',
                '--baseline',str(frozen),'--opentpu-root',str(home/'FeSens-openTPU-qwen'),
                '--compiler',str(home/'LPNDialect-heartbeat/build/tools/lpn-compile/lpn-compile'),
                '--output',str(dest/'replay')]
            try:
                started=time.perf_counter()
                with (dest/'run.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    item={'status':'running','pid':child.pid,'command':command,'started_unix':time.time()}
                    state['cases'][name]=item;state['status']='replaying';atomic(state_path,state)
                    code=child.wait()
                elapsed=time.perf_counter()-started
                item.update(returncode=code,ended_unix=time.time(),external_process_seconds=elapsed)
                if code!=0:raise RuntimeError('Replay process failed; inspect '+str(dest/'run.log'))
                measured=json.loads((dest/'replay/report.json').read_text())
                assert measured['complete_generation_replayed']
                assert measured['all_inputs_identical'] and measured['all_logits_identical'] and measured['all_predicted_cycles_identical']
                assert measured['generated_tokens']==report['generated_tokens']
                item.update(status='complete',steps=measured['steps'],
                    session_seconds=measured['session_seconds'],
                    lpn_core_seconds=measured['lpn_core_seconds'],
                    full_lpn_seconds=measured['single_use_lpn_seconds'],
                    recorded_rtl_core_seconds=measured['recorded_rtl_core_seconds'],
                    complete_process_speedup_vs_rtl_core=measured['recorded_rtl_core_seconds']/elapsed)
                # Regenerable input dumps otherwise consume ~1.7 GiB per case.
                shutil.rmtree(dest/'replay/current')
            except Exception:
                state['cases'].setdefault(name,{}).update(status='failed',error=traceback.format_exc())
            print(json.dumps({'case':name,**state['cases'][name]}),flush=True)
            atomic(state_path,state)
        state['status']='waiting_for_complete_cases';state['updated_unix']=time.time()
        if all(c['status'] in ('complete','failed') for c in batch['jobs']) and len(state['cases'])==len(batch['jobs']):
            state['status']='complete'
            state['all_replays_passed']=all(c['status']=='complete' for c in state['cases'].values())
            atomic(state_path,state);break
        atomic(state_path,state);time.sleep(30)

if __name__=='__main__':main()
