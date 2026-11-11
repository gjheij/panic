"""Parent-owned checkpoint progress, including explicit center failures."""
import csv
import json
import os
import tempfile
import time
from .fit_guard import CenterFitFailure, SearchlightFitError


class CenterProgress:
    def __init__(self, results, checkpoint, policy, base, update_interval=1000, save_seconds=60):
        if not isinstance(save_seconds, (int,float)) or isinstance(save_seconds,bool) or not 0 < save_seconds < float('inf'):
            raise ValueError('parallel.checkpoint_interval_seconds must be positive and finite')
        self.results, self.checkpoint, self.policy, self.base = results, checkpoint, policy, base
        self.failures = checkpoint.failures if checkpoint is not None else {}
        self.update_interval = update_interval
        self.save_seconds = save_seconds
        self.last_save = time.monotonic()
        self._completed = sum(row is not None or str(i) in self.accepted_failures for i,row in enumerate(results))

    @property
    def accepted_failures(self):
        return self.failures if self.policy['enabled'] and self.policy['on_failure']=='mark_center' else {}

    @property
    def pending(self):
        return [i for i,row in enumerate(self.results) if row is None and str(i) not in self.accepted_failures]

    @property
    def completed(self):
        return self._completed

    def flush(self):
        if self.checkpoint is not None:
            self.checkpoint.save()
        if self.failures:
            path = self.base+'_fit_failures.csv'
            os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
            fd,temp=tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),prefix='.checkpoint-fit-failures-')
            try:
                with os.fdopen(fd,'w',newline='') as stream:
                    fields=['center_index','x','y','z','reason_code','phase','permutation','perm_seed','error','diagnostics']
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for index, entry in sorted(self.failures.items(),key=lambda item:int(item[0])):
                        x,y,z=entry['center']
                        writer.writerow(dict(center_index=int(index),x=x,y=y,z=z,
                            reason_code=entry.get('reason_code','solver_nonconvergence'),phase=entry.get('phase'),
                            permutation=entry.get('permutation'),perm_seed=entry.get('perm_seed'),error=entry.get('error'),
                            diagnostics=json.dumps(entry,allow_nan=False)))
                    stream.flush();os.fsync(stream.fileno())
                os.replace(temp,path)
            finally:
                if os.path.exists(temp):os.unlink(temp)
        else:
            try:os.unlink(self.base+'_fit_failures.csv')
            except FileNotFoundError:pass
        self.last_save=time.monotonic()

    def record(self,index,result):
        before = self.results[index] is not None or str(index) in self.accepted_failures
        if isinstance(result,CenterFitFailure):
            self.failures[str(index)]=dict(result.diagnostics,center=list(result.center))
            self.results[index]=None
            self._completed += int(str(index) in self.accepted_failures)-int(before)
            self.flush()
            if self.policy['on_failure']=='raise':
                raise SearchlightFitError('Searchlight center {} failed convergence; checkpoint retained'.format(index),self.failures[str(index)])
            return self.checkpoint is not None
        self.results[index]=result
        self.failures.pop(str(index),None)
        self._completed += 1-int(before)
        completed=self.completed
        if self.checkpoint is not None and (completed==len(self.results) or
                self.update_interval>0 and completed%self.update_interval==0 or
                time.monotonic()-self.last_save>=self.save_seconds):
            self.flush()
            return True
        return False

    def output_rows(self):
        for index,row in enumerate(self.results):
            if row is not None:
                yield tuple(row)+(None,) if len(row)==9 else row
            elif str(index) in self.accepted_failures:
                entry=self.accepted_failures[str(index)]
                # Invalidate the WHOLE center, including its observed score.
                # A truncated permutation null must never yield a p-value.
                yield (tuple(entry['center']),float('nan'),float('nan'),float('nan'),float('nan'),
                       int(entry.get('n_features',0)),0,0,0,None)
            else:
                raise RuntimeError('Searchlight center {} has no result'.format(index))
