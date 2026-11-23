"""Synthetic-image and algebra tests; no real subject data required."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_4d_images_feature_order_and_run_centering(monkeypatch, tmp_path):
    reader = module_from_path('residuals_synthetic', ROOT / 'panic' / 'residuals.py')
    class Img:
        def __init__(self, data, affine=None):
            self.dataobj = data
            self.shape = data.shape
            self.affine = np.eye(4) if affine is None else affine
    # Mock only the NIfTI IO layer; use real 4D spatial/time arrays.
    images = {}
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda p: images[p]))
    paths = []
    for run in range(2):
        p = tmp_path / f'run{run}.nii.gz'
        p.touch()
        data = np.arange(2*3*2*5).reshape(2, 3, 2, 5).astype(float) + run*100
        images[str(p)] = Img(data)
        paths.append(p)
    ref = Img(np.zeros((2,3,2,8)))
    idx = np.array([0, 4, 11])
    R, groups = reader.load_roi_residuals(paths, ref, idx)
    assert R.shape == (10, 3)
    assert np.array_equal(groups, [0]*5 + [1]*5)
    expected = images[str(paths[0])].dataobj.reshape(12,5)[idx].T
    np.testing.assert_allclose(R[:5], expected - expected.mean(axis=0))
    np.testing.assert_allclose(R[5:].mean(axis=0), 0, atol=1e-12)
    images[str(paths[1])].affine[0,3] = 2
    with pytest.raises(ValueError, match='grid/affine'):
        reader.load_roi_residuals(paths, ref, idx)


def load_rsa_with_stub(monkeypatch):
    # Avoid requiring the optional neuroimaging stack for pure algebra tests.
    panic = types.ModuleType('panic'); panic.__path__ = []
    utils = types.ModuleType('panic.utils')
    utils.load_feature_matrix = lambda x, cols=None: np.asarray(x) if cols is None else np.asarray(x)[:,cols]
    monkeypatch.setitem(sys.modules, 'panic', panic)
    monkeypatch.setitem(sys.modules, 'panic.utils', utils)
    return module_from_path('rsa_synthetic', ROOT / 'panic' / 'plugins' / 'rsa.py')


def test_crossnobis_uses_supplied_residual_covariance(monkeypatch):
    rsa = load_rsa_with_stub(monkeypatch)
    # 3 runs, 2 conditions, 2 features; no within-run label imbalance.
    X = np.array([[2,1],[0,0], [3,2],[0,0], [4,3],[0,0]], float)
    y = np.array([0,1]*3)
    g = np.repeat([0,1,2],2)
    residuals = np.random.default_rng(123).normal(size=(100,2)) * [1,4]
    a = rsa.crossnobis_rdm(X,y,g,[0,1],noise_normalization='identity')[0]
    b = rsa.crossnobis_rdm(X,y,g,[0,1],noise_residuals=residuals,
                           noise_normalization='diagonal')[0]
    assert np.isfinite(a[0,1]) and np.isfinite(b[0,1])
    assert not np.isclose(a[0,1],b[0,1])
    with pytest.raises(ValueError, match='requires'):
        rsa.crossnobis_rdm(X,y,g,[0,1],noise_normalization='shrinkage')
    c = rsa.crossnobis_rdm(X,y,g,[0,1],noise_residuals=residuals,
                           noise_normalization='shrinkage')[0]
    assert np.isfinite(c[0,1])


def test_missing_condition_run_pair_and_negative_distance(monkeypatch):
    rsa = load_rsa_with_stub(monkeypatch)
    X = np.array([[1.],[0.],[2.],[0.],[1.]], float)
    y = np.array([0,1,0,1,0]); g = np.array([0,0,1,1,2])
    rdm, counts, trials, _ = rsa.crossnobis_rdm(X,y,g,[0,1])
    assert counts[0,1] == 2
    assert trials[2,1] == 0
    assert rdm[0,1] == pytest.approx(2.)
    # Opposite-signed independent run contrasts yield negative distances.
    X2 = np.array([[1.],[0.],[-1.],[0.]])
    rdm2,_,_,_ = rsa.crossnobis_rdm(X2,np.array([0,1,0,1]),
                                     np.array([0,0,1,1]),[0,1])
    assert rdm2[0,1] == pytest.approx(-1.)
