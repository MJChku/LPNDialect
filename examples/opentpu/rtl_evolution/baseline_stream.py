"""Read completed original-RTL observations, optionally from a live local job."""
import json
import os
from pathlib import Path
import time


class BaselineReference:
    def __init__(self,report,job=None,wait_seconds=28800):
        self.report=Path(report)
        self.job=Path(job) if job is not None else None
        self.wait_seconds=wait_seconds
        if self.report.exists():
            self.info=self._completed()
        elif self.job is not None:
            manifest=json.loads(self.job.read_text())
            command=manifest['command']
            identity=json.loads((self.report.parent/'identity.json').read_text())
            model=command[command.index('--model-dir')+1]
            prompt_path=Path(command[command.index('--prompt-json')+1])
            prompt=json.loads(prompt_path.read_text())['ids']
            self.info=dict(config=identity['config'],model_directory=model,
                           prompt_ids=prompt,prompt_tokens=len(prompt),requested_output_tokens=32)
            self._check_owner()
        else:
            raise ValueError('completed baseline report or --baseline-job is required')
        if self.info['prompt_tokens']!=128:
            raise ValueError('expected a 128-token baseline prompt')

    def _completed(self):
        report=json.loads(self.report.read_text())
        if report.get('status')!='complete' or report.get('mode')!='validate-rtl':
            raise ValueError('baseline report is not a completed actual-RTL observation')
        if report['prompt_tokens']!=128 or report['returned_output_tokens']!=32:
            raise ValueError('baseline did not finish the requested 128/32 case')
        return report

    def _check_owner(self):
        if self.job is None:
            raise RuntimeError('baseline observation is missing and no live job was specified')
        job=json.loads(self.job.read_text())
        if job['status']!='running':
            if self.report.exists():
                self._completed()
                return
            raise RuntimeError(f"baseline job is {job['status']}; missing observation cannot be reused")
        pid=job['pid']
        try:
            os.kill(pid,0)
            # Refuse a stale PID or a zombie rather than treating it as progress.
            process=Path(f'/proc/{pid}')
            if process.is_dir():
                if process.joinpath('stat').read_text().split(') ',1)[1].startswith('Z '):
                    raise ProcessLookupError(pid)
                command=process.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                if [x.decode() for x in command]!=job['command']:
                    raise RuntimeError('baseline PID belongs to a different command')
        except (ProcessLookupError,FileNotFoundError) as error:
            if self.report.exists():
                self._completed()
                return
            raise RuntimeError(f'baseline worker {pid} is no longer live') from error

    def _wait(self,read,description):
        start=time.monotonic();last_notice=0
        while True:
            value=read()
            if value is not None:return value
            # A final report may have appeared between the read and owner check.
            if self.report.exists():
                self._completed()
                value=read()
                if value is not None:return value
                raise RuntimeError(f'completed baseline lacks {description}')
            self._check_owner()
            elapsed=time.monotonic()-start
            if elapsed>self.wait_seconds:
                raise TimeoutError(f'timed out waiting for {description} from the baseline')
            if elapsed>=last_notice:
                print(json.dumps(dict(stage='waiting_for_baseline',observation=description,
                                      seconds=round(elapsed,1))),flush=True)
                last_notice=elapsed+30
            time.sleep(2)

    def kernel(self,index):
        def read():
            if self.report.exists():
                records=self._completed()['kernels']
                if index<len(records):return records[index]
                return None
            path=self.report.parent/'kernels'/f'{index:04d}'/'record.json'
            return json.loads(path.read_text()) if path.exists() else None
        record=self._wait(read,f'kernel {index}')
        if record['step']!=index or not record['all_dram_and_tmem_bit_exact'] or not record['identical_input_files']:
            raise RuntimeError(f'kernel {index} lacks verified original-RTL evidence')
        return record

    def complete(self):
        return self._wait(lambda:self._completed() if self.report.exists() else None,'final 128/32 report')
