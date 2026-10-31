"""Exercise the real shared-table writer without importing PANIC's imaging stack."""
import argparse
import ast
import contextlib
import io
import json
import importlib.util
import multiprocessing
from pathlib import Path
import tempfile
import unittest

import pandas as pd

FILE = Path(__file__).resolve().parents[1] / 'panic' / 'result_store.py'
spec = importlib.util.spec_from_file_location('panic_result_store', FILE)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


def row(method='LSA', roi='bl', identity='first', status='ok'):
    return dict(subject='sub-017', analysis='contrast', analysis_type='scikit_decoding',
                source='halfpipe', method=method, roi=roi, hemi='left', analysis_id=identity,
                batch_job_id=identity, observed_acc=.75 if status == 'ok' else float('nan'), status=status)


def concurrent_write(folder, method):
    result = row(method=method, identity=method)
    store.merge_result_tables(Path(folder)/'results.csv', Path(folder)/'null.csv',
                              pd.DataFrame([result]), pd.DataFrame([{**result, 'perm': 0, 'acc': .5}]),
                              overwrite=True)


class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, rows, overwrite=True, n=3):
        draws = [{**r, 'perm': i, 'acc': .5} for r in rows if r['status'] == 'ok' for i in range(n)]
        return store.merge_result_tables(self.root/'results.csv', self.root/'null.csv',
                                         pd.DataFrame(rows), pd.DataFrame(draws), overwrite=overwrite)

    def test_lsa_lss_are_concatenated_and_overwrite_replaces_only_current_cells(self):
        self.write([row('LSA'), row('LSA', roi='lat')])
        self.write([row('LSS')])
        self.write([row('LSA', identity='replacement')], n=1)
        rows = pd.read_csv(self.root/'results.csv', dtype=str)
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows['method']), {'LSA','LSS'})
        self.assertEqual(rows.loc[(rows.method=='LSA') & (rows.roi=='bl'),'analysis_id'].item(), 'replacement')
        draws = pd.read_csv(self.root/'null.csv', dtype=str)
        self.assertEqual(len(draws), 7)  # 3 lat + 3 LSS + 1 replaced LSA

    def test_incremental_same_cell_is_deduplicated_but_other_analysis_ids_survive(self):
        self.write([row()], overwrite=False)
        self.write([row()], overwrite=False)
        self.write([row(identity='other')], overwrite=False)
        self.assertEqual(len(pd.read_csv(self.root/'results.csv')), 2)
        self.assertEqual(len(pd.read_csv(self.root/'null.csv')), 6)

    def test_skip_removes_obsolete_null_draws_without_touching_other_model(self):
        self.write([row('LSA'), row('LSS')])
        self.write([row('LSA', status='skipped')])
        results = pd.read_csv(self.root/'results.csv')
        self.assertEqual(set(results.status), {'ok', 'skipped'})
        self.assertEqual(set(pd.read_csv(self.root/'null.csv').method), {'LSS'})

    def test_empty_incremental_update_preserves_both_tables(self):
        self.write([row()])
        before = [(self.root/name).read_bytes() for name in ['results.csv', 'null.csv']]
        self.write([], overwrite=False)
        self.assertEqual(before, [(self.root/name).read_bytes() for name in ['results.csv', 'null.csv']])

    def test_processes_do_not_lose_each_others_rows(self):
        context = multiprocessing.get_context('fork')
        processes = [context.Process(target=concurrent_write, args=(str(self.root), f'M{i}')) for i in range(12)]
        for process in processes: process.start()
        for process in processes:
            process.join(15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(len(pd.read_csv(self.root/'results.csv')), 12)
        self.assertEqual(len(pd.read_csv(self.root/'null.csv')), 12)

    def test_cli_exposes_shared_output_capabilities(self):
        tree=ast.parse((FILE.parent/'cli.py').read_text())
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('action_capabilities','build_parser')]
        namespace={'argparse':argparse,'json':json,'action_show':lambda args:None,
                   'action_config':lambda args:None,'action_run':lambda args:None}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(FILE.parent/'cli.py'),'exec'),namespace)
        args=namespace['build_parser']().parse_args(['capabilities'])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):args.func(args)
        self.assertEqual(json.loads(output.getvalue()),{'shared_roi_tables':1,'batch_job_id':1})


if __name__ == '__main__': unittest.main()
