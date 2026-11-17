"""Destination-aware CSV publication, including simulated ENOSPC on close."""
import ast
import errno
import importlib
import importlib.util
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('csv_atomic_store', ROOT/'panic/result_store.py')
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


def full_disk():
    return OSError(errno.ENOSPC, 'No space left on device')


class FailingStream:
    def __init__(self, wrapped, stage):
        self.wrapped, self.stage = wrapped, stage

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.wrapped.close()
        if self.stage == 'close':
            raise full_disk()

    def flush(self):
        if self.stage == 'flush':
            raise full_disk()
        return self.wrapped.flush()


class AtomicCSVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'results.csv'
        self.previous = b'previous,valid\n1,2\n'
        self.path.write_bytes(self.previous)
        self.frame = pd.DataFrame({'value': [1, 2], 'name': ['caf\u00e9', 'two']})

    def unchanged(self):
        self.assertEqual(self.path.read_bytes(), self.previous)
        self.assertFalse(list(self.root.glob('.csv-write-*')))

    def assert_io_error(self, operation, stage):
        with self.assertLogs(store.logger, level=logging.INFO) as log:
            with self.assertRaises(OSError) as caught:
                operation()
        error = caught.exception
        self.assertEqual(error.errno, errno.ENOSPC)
        self.assertEqual(error.filename, str(self.path.absolute()))
        self.assertIn(stage, str(error))
        self.assertIn('temporary=', str(error))
        self.assertIsInstance(error.__cause__, OSError)
        self.assertIn('Writing CSV: '+str(self.path), log.output[0])
        self.unchanged()

    def test_success_bytes_permissions_and_sibling_temporary(self):
        self.path.chmod(0o640)
        replace = store.os.replace
        calls = []
        def publish(source, destination):
            calls.append((Path(source), Path(destination)))
            self.assertEqual(self.path.read_bytes(), self.previous)
            return replace(source, destination)
        with patch.object(store.os, 'replace', side_effect=publish):
            store.write_dataframe_csv(self.frame, self.path, index=False)
        self.assertEqual(self.path.read_text(), self.frame.to_csv(index=False))
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)
        self.assertEqual(calls[0][0].parent, self.path.parent)
        self.assertFalse(calls[0][0].exists())

    def test_temporary_creation_failure_logs_destination_first(self):
        with patch.object(store.tempfile, 'mkstemp', side_effect=full_disk()):
            self.assert_io_error(lambda: store.write_dataframe_csv(self.frame, self.path, index=False), 'create temporary file')

    def test_partial_serialization_failure_preserves_existing(self):
        class Frame:
            def to_csv(self, stream, **kwargs):
                stream.write('partial,data\n')
                raise full_disk()
        self.assert_io_error(lambda: store.write_dataframe_csv(Frame(), self.path), 'write data')

    def test_flush_failure_preserves_existing(self):
        original = store.os.fdopen
        with patch.object(store.os, 'fdopen', side_effect=lambda *a, **kw: FailingStream(original(*a, **kw), 'flush')):
            self.assert_io_error(lambda: store.write_dataframe_csv(self.frame, self.path, index=False), 'flush data')

    def test_close_failure_preserves_existing(self):
        original = store.os.fdopen
        with patch.object(store.os, 'fdopen', side_effect=lambda *a, **kw: FailingStream(original(*a, **kw), 'close')):
            self.assert_io_error(lambda: store.write_dataframe_csv(self.frame, self.path, index=False), 'close temporary file')

    def test_fsync_failure_preserves_existing(self):
        with patch.object(store.os, 'fsync', side_effect=full_disk()):
            self.assert_io_error(lambda: store.write_dataframe_csv(self.frame, self.path, index=False), 'fsync data')

    def test_replace_failure_preserves_existing(self):
        with patch.object(store.os, 'replace', side_effect=full_disk()):
            self.assert_io_error(lambda: store.write_dataframe_csv(self.frame, self.path, index=False), 'replace destination')

    def test_failed_new_output_is_absent(self):
        self.path.unlink()
        with patch.object(store.os, 'fsync', side_effect=full_disk()), self.assertRaises(OSError):
            store.write_dataframe_csv(self.frame, self.path, index=False)
        self.assertFalse(self.path.exists())
        self.assertFalse(list(self.root.iterdir()))

    def test_non_io_serialization_error_preserved(self):
        frame = Mock()
        frame.to_csv.side_effect = ValueError('invalid serialization')
        with self.assertRaisesRegex(ValueError, 'invalid serialization'):
            store.write_dataframe_csv(frame, self.path)
        self.unchanged()

    def test_cleanup_error_does_not_mask_primary_error(self):
        unlink = store.os.unlink
        with patch.object(store.os, 'fsync', side_effect=full_disk()), \
             patch.object(store.os, 'unlink', side_effect=PermissionError('cleanup denied')):
            with self.assertRaises(OSError) as caught:
                store.write_dataframe_csv(self.frame, self.path, index=False)
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        self.assertIn('fsync data', str(caught.exception))
        self.assertEqual(self.path.read_bytes(), self.previous)
        for path in self.root.glob('.csv-write-*'):
            unlink(path)

    def test_rows_and_qc_writer_use_protected_publication(self):
        with patch.object(store.os, 'fsync', side_effect=full_disk()):
            self.assert_io_error(lambda: store._atomic(self.path, ['x'], [{'x': 1}]), 'fsync data')
        store.write_qc_rows(self.path, [{'subject': 'a', 'roi': 'bl', 'hemi': 'left', 'tsnr_mean': 2}])
        self.assertIn('tsnr_mean', self.path.read_text())

    def test_log_storage_failure_keeps_destination_context(self):
        with patch.object(store.logger, 'info', side_effect=full_disk()), \
             patch.object(store.logger, 'error', side_effect=full_disk()):
            with self.assertRaises(OSError) as caught:
                store.write_dataframe_csv(self.frame, self.path, index=False)
        self.assertEqual(caught.exception.filename, str(self.path))
        self.assertIn('log destination', str(caught.exception))
        self.unchanged()

    def test_second_table_failure_keeps_old_null_file(self):
        null_path = self.root/'null.csv'
        null_path.write_bytes(b'subject,acc\nprevious,0.5\n')
        previous_null = null_path.read_bytes()
        replace = store.os.replace
        def publish(source, destination):
            if Path(destination) == null_path:
                raise full_disk()
            return replace(source, destination)
        with patch.object(store.os, 'replace', side_effect=publish), self.assertRaises(OSError) as caught:
            store.merge_result_tables(self.path, null_path,
                pd.DataFrame([{'subject':'new','observed_acc':.75}]),
                pd.DataFrame([{'subject':'new','perm':0,'acc':.5}]))
        self.assertEqual(caught.exception.filename, str(null_path))
        self.assertEqual(null_path.read_bytes(), previous_null)
        self.assertIn('new', self.path.read_text())  # The pair is not one transaction.
        self.assertFalse(list(self.root.glob('.csv-write-*')))

    def test_tsv_preserves_pandas_format(self):
        path = self.root/'events.tsv'
        store.write_dataframe_csv(self.frame, path, index=False, sep='\t')
        self.assertEqual(path.read_text(), self.frame.to_csv(index=False, sep='\t'))

    def test_searchlight_failure_csv_preserved_on_error(self):
        package = 'panic_atomic_csv_test'
        pkg = types.ModuleType(package); pkg.__path__ = [str(ROOT/'panic')]
        with patch.dict(sys.modules, {package: pkg}):
            progress = importlib.import_module(package+'.searchlight_progress')
        base = self.root/'searchlight'
        target = Path(str(base)+'_fit_failures.csv')
        target.write_bytes(self.previous)
        checkpoint = types.SimpleNamespace(failures={'0': {'center': [0,0,0], 'error': 'fit failed'}}, save=lambda: None)
        runner = progress.CenterProgress([None], checkpoint, {'enabled':True,'on_failure':'mark_center'}, str(base))
        with patch.object(store.os, 'fsync', side_effect=full_disk()), self.assertRaises(OSError) as caught:
            runner.flush()
        self.assertEqual(caught.exception.filename, str(target))
        self.assertEqual(target.read_bytes(), self.previous)
        self.assertFalse(list(self.root.glob('.checkpoint-fit-failures-*')))

    def test_actual_trial_order_export_fragment(self):
        tree = ast.parse((ROOT/'panic/decode.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ClassifySubject')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_init_betas')
        start = next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Attribute) and t.attr=='trial_order_file' for t in n.targets))
        decoder = types.SimpleNamespace(gen_settings={'method':'LSA','source':'halfpipe','batch_job_id':'abc123'},
            searchlight=False,events_df=self.frame,generate_filename=lambda **kw:str(self.path))
        namespace = {'self':decoder,'write_dataframe_csv':store.write_dataframe_csv,'logger':Mock()}
        with patch.object(store.os, 'fsync', side_effect=full_disk()):
            with self.assertRaises(OSError) as caught:
                exec(compile(ast.Module(body=init.body[start:],type_ignores=[]),'trial_order_fragment','exec'),namespace)
        self.assertEqual(caught.exception.filename, str(self.path))
        self.unchanged()

    def test_simulation_events_use_atomic_tsv(self):
        tree = ast.parse((ROOT/'panic/simulate.py').read_text())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='condition_file')
        namespace = {'pd':pd, 'Path':Path, 'write_dataframe_csv':store.write_dataframe_csv}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'simulation_events','exec'),namespace)
        path = namespace['condition_file'](self.root,self.frame)
        self.assertEqual(path.read_text(), self.frame.to_csv(index=False,sep='\t'))


if __name__ == '__main__':
    unittest.main()
