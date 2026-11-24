"""Standalone tests, avoiding PANIC's optional imaging imports."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Isolated module namespace; do not import the imaging stack.
for name in ('panic', 'panic.plugins'):
    sys.modules[name] = types.ModuleType(name)
utils = types.ModuleType('panic.utils')
utils.load_feature_matrix = lambda X, *a, **kw: np.asarray(X)
sys.modules['panic.utils'] = utils
r = load('panic.plugins.result', 'panic/plugins/result.py')
s = load('panic.result_store', 'panic/result_store.py')
rsa = load('panic.plugins.rsa', 'panic/plugins/rsa.py')

class Tests(unittest.TestCase):
    def test_rows(self):
        p = r.normalize_plugin_result(r.PluginResult(.4, rows=[{'model':'a','observed':.4}, {'model':'b','observed':.2}], row_keys=('model',), primary_key=('a',)))
        base = dict(subject='sub-01', observed_acc=.4, null_mean=.1, delta=.3, p_value=.02, n_perms=100, n_run=100)
        d = dict(plugin_rows=p.rows, plugin_row_keys=p.row_keys, plugin_primary_key=p.primary_key)
        a,b = s.expand_result_rows(base,d)
        self.assertEqual(a['p_value'],.02)
        self.assertTrue(np.isnan(b['p_value']))
        self.assertEqual(b['observed_acc'],.2)
        self.assertEqual(s.primary_row_identity(d),{'model':'a'})
        self.assertEqual(len(s.expand_result_rows(base,{})),1)
        p.rows[1]['subject']='wrong'
        with self.assertRaises(ValueError): s.expand_result_rows(base,d)
        p.rows[1]['model']='a'
        with self.assertRaises(ValueError): r.normalize_plugin_result(p)
    def test_legacy(self):
        self.assertEqual(r.normalize_plugin_result(.5).score,.5)
        self.assertEqual(r.normalize_plugin_result((.5,{'a':1})).artifacts,{'a':1})
    def test_merge(self):
        with TemporaryDirectory() as t:
            a,b = Path(t)/'results.csv',Path(t)/'null.csv'
            base=dict(subject='sub-01',analysis='rsa',analysis_type='rsa',source='stglm',method='m',roi='lat',hemi='left',analysis_id='id')
            frame=pd.DataFrame([dict(base,model='a'),dict(base,model='b'),dict(base,roi='bl',model='a')])
            s.merge_result_tables(a,b,frame,pd.DataFrame())
            merged=s.merge_result_tables(a,b,pd.DataFrame([dict(base,model='c')]),pd.DataFrame())
            self.assertEqual(set(merged.model),{'a','c'})
            self.assertEqual(len(merged),2)
    def test_rsa(self):
        rng=np.random.default_rng(2)
        X=rng.normal(size=(18,5)); y=np.tile(np.repeat(np.arange(3),2),3); g=np.repeat([1,3,4],6)
        m=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])
        kw=dict(groups=g,metric='crossnobis',model_rdms={'a':m,'b':m**2},target_model='a')
        p=r.normalize_plugin_result(rsa.rsa_plugin(X,y,{},return_artifacts=True,**kw))
        self.assertEqual(len(p.rows),2)
        np.testing.assert_equal(p.artifacts['rsa_model_scores'],[x['observed'] for x in p.rows])
        self.assertEqual(p.score,rsa.rsa_plugin(X,y,{},return_artifacts=False,**kw))

if __name__=='__main__': unittest.main()
