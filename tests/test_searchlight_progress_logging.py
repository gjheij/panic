"""Useful progress summaries at timed checkpoints, without repeated path noise."""
import importlib.util
import itertools
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('checkpoint_progress_logging',ROOT/'panic/checkpoint.py')
checkpoint=importlib.util.module_from_spec(spec);spec.loader.exec_module(checkpoint)


class ProgressLoggingTests(unittest.TestCase):
    def test_checkpoint_destination_info_once_debug_afterwards(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.joblib'
            obj=checkpoint.SearchlightCheckpoint(str(path),'id',[0],1)
            with self.assertLogs(checkpoint.logger,level='DEBUG') as logs:
                obj.save();obj.save()
            self.assertEqual(sum(line.startswith('INFO:') for line in logs.output),1)
            self.assertEqual(sum(line.startswith('DEBUG:') for line in logs.output),1)
            self.assertTrue(all(str(path) in line for line in logs.output))

    def test_subsequent_save_error_still_reports_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.joblib'
            obj=checkpoint.SearchlightCheckpoint(str(path),'id',[0],1);obj.save()
            with self.assertLogs(checkpoint.logger,level='INFO') as logs, \
                 patch.object(checkpoint.os,'fsync',side_effect=OSError(28,'No space left on device')),self.assertRaises(OSError):
                obj.save()
            self.assertTrue(any('ERROR:' in line and str(path) in line and 'fsync' in line for line in logs.output))

    def test_timed_save_emits_progress_below_center_interval(self):
        import test_svc_fit_guard as fixture
        case=fixture.SearchlightIntegrationTests();case.setUp()
        original=fixture.sp.CenterProgress.record
        def record(progress,*args,**kwargs):
            progress.last_save-=61
            return original(progress,*args,**kwargs)
        try:
            with patch.object(fixture.sp.CenterProgress,'record',record),self.assertLogs(case.ns['logger'],level='INFO') as logs:
                case.run_searchlight()
            first=next(line for line in logs.output if 'Progress: 1/4' in line)
            for field in ('25.0%', 'centers/s', 'checkpoint saved', 'elapsed', 'ETA'):
                self.assertIn(field,first)
        finally:case.doCleanups()

    def test_time_progress_without_checkpoint_has_no_saved_claim(self):
        import test_svc_fit_guard as fixture
        case=fixture.SearchlightIntegrationTests();case.setUp()
        case.cfg['searchlight']['checkpoint']=False
        clock=itertools.count(0,61)
        case.ns['time']=types.SimpleNamespace(monotonic=lambda:next(clock))
        try:
            with self.assertLogs(case.ns['logger'],level='INFO') as logs:case.run_searchlight()
            lines=[line for line in logs.output if 'Progress:' in line]
            self.assertTrue(any('1/4' in line for line in lines))
            self.assertFalse(any('checkpoint saved' in line for line in lines))
        finally:case.doCleanups()


if __name__=='__main__':unittest.main()
