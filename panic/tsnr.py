"""Optional run-matched tSNR QC on the final decoding ROI grid."""
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

logger = logging.getLogger(__name__)
ENTITY_KEYS = ('sub', 'ses', 'task', 'acq', 'run', 'echo', 'dir', 'rec')


def acquisition_entities(path):
    path = Path(path)
    entities = {}
    for token in path.name.split('_'):
        if '-' in token:
            key, value = token.split('-', 1)
            if key in ENTITY_KEYS or key in ('space', 'desc'):
                entities[key] = value
    for part in path.parts[:-1]:
        if part.startswith(('sub-', 'ses-')):
            key, value = part.split('-', 1)
            entities.setdefault(key, value)
    if entities.get('run', '').isdigit():
        entities['run'] = str(int(entities['run']))
    return entities


def tsnr_sidecar(path):
    path = Path(path)
    return path.with_name(path.name.removesuffix('.nii.gz') + '.json')


def tsnr_metadata(path):
    sidecar = tsnr_sidecar(path)
    if not sidecar.exists():
        return {}
    try:
        value = json.loads(sidecar.read_text())
        if not isinstance(value, dict):
            raise ValueError('sidecar must contain a JSON object')
        return value
    except (OSError, ValueError) as error:
        return {'_metadata_error': str(error)}


def tsnr_space(path, metadata):
    """Registration SpatialReference supersedes inherited source settings."""
    if metadata.get('_metadata_error'):
        return None
    def normalize(value):
        if value in ('native', 'T1w'):
            return 'T1w'
        # 'standard' alone does not identify a particular template.
        return value if isinstance(value, str) and value not in ('standard', '') else None
    named = normalize(acquisition_entities(path).get('space'))
    reference = normalize(metadata.get('SpatialReference'))
    settings = metadata.get('Setting', metadata.get('setting', {}))
    configured = normalize(settings.get('Space', settings.get('space'))) if isinstance(settings, dict) else None
    if 'SpatialReference' in metadata and reference is None:
        return None
    if named and reference and named != reference:
        return None
    if not reference and named and configured and named != configured:
        return None
    return reference or named or configured


def matching_tsnr_maps(beta_files, candidates):
    """Match acquisitions and known coordinate spaces; never infer space from a grid."""
    units = {}
    for path in beta_files:
        key = tuple(acquisition_entities(path).get(k) for k in ENTITY_KEYS)
        units.setdefault(key, []).append(str(path))
    records = []
    for key, betas in sorted(units.items(), key=lambda pair: tuple(v or '' for v in pair[0])):
        spaces = {tsnr_space(p, tsnr_metadata(p)) for p in betas}
        matches, registered, rejected = [], [], []
        error = ''
        if len(spaces) != 1 or None in spaces:
            error = 'Beta coordinate space is unknown or inconsistent; supply native/T1w beta metadata'
        else:
            space = next(iter(spaces))
            for path in sorted(set(map(str, candidates))):
                if tuple(acquisition_entities(path).get(k) for k in ENTITY_KEYS) != key:
                    continue
                metadata = tsnr_metadata(path)
                if tsnr_space(path, metadata) != space:
                    rejected.append(path)
                    continue
                if acquisition_entities(path).get('desc') == 'betagrid':
                    ref = metadata.get('ReferenceImage')
                    grid = metadata.get('ReferenceGrid')
                    if (metadata.get('SpatialReference') != space or not isinstance(ref, str)
                            or tuple(acquisition_entities(ref).get(k) for k in ENTITY_KEYS) != key
                            or not isinstance(grid, dict) or not {'Shape', 'Affine'}.issubset(grid)):
                        rejected.append(path)
                        continue
                    registered.append(path)
                else:
                    matches.append(path)
            matches = registered or matches
            if not matches and rejected:
                error = 'No map with matching, verified coordinate-space metadata and registration provenance'
        records.append({'entities': dict(zip(ENTITY_KEYS, key)), 'files': matches,
                        'betas': betas, 'error': error})
    return records


def resample_values(data, affine, target_affine, target_shape):
    """Linear interpolation with full finite-positive support; no padded zeros."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or not np.isfinite(affine).all() or not np.isfinite(target_affine).all():
        raise ValueError('tSNR maps must be 3D with finite affines')
    valid = np.isfinite(data) & (data > 0)
    values = np.where(valid, data, 0.0)
    if tuple(data.shape) == tuple(target_shape) and np.allclose(affine, target_affine):
        return values, valid
    transform = np.linalg.inv(affine) @ target_affine
    transform[np.abs(transform) < 1e-12] = 0
    def sample(array):
        return affine_transform(array, transform[:3,:3], offset=transform[:3,3],
            output_shape=tuple(target_shape), order=1, mode='constant', cval=0.0, prefilter=False)
    support = sample(valid.astype(float))
    values = sample(values)
    return values, (support >= 1 - 1e-6) & np.isfinite(values) & (values > 0)


class TsnrQC:
    """Cache map resampling once per beta grid, then summarize each final ROI."""
    def __init__(self, beta_files, candidates, loader=None):
        self.units = matching_tsnr_maps(beta_files, candidates)
        self.loader = loader
        self.cache = {}
        self.metadata = {path: tsnr_metadata(path) for unit in self.units for path in unit["files"]}

    def _load(self, path, affine, shape, unit):
        key = (path, tuple(shape), tuple(np.asarray(affine).ravel()))
        if key not in self.cache:
            try:
                if self.loader is None:
                    import nibabel as nib
                    img = nib.load(path)
                else:
                    img = self.loader(path)
                if acquisition_entities(path).get('desc') == 'betagrid':
                    reference = tsnr_metadata(path)['ReferenceGrid']
                    def matches_grid(image):
                        return (list(image.shape[:3]) == reference['Shape']
                                and np.asarray(reference['Affine']).shape == (4, 4)
                                and np.allclose(image.affine, reference['Affine'], rtol=0, atol=1e-4))
                    if not matches_grid(img):
                        raise ValueError('Registered tSNR image does not match its ReferenceGrid')
                    for beta in unit['betas']:
                        beta_img = self.loader(beta) if self.loader else nib.load(beta)
                        if not matches_grid(beta_img):
                            raise ValueError('Registered tSNR ReferenceGrid does not match the selected beta grid')
                self.cache[key] = (*resample_values(img.get_fdata(), img.affine, affine, shape), '')
            except Exception as error:
                logger.warning('tSNR QC could not read %s: %s', path, error)
                self.cache[key] = (None, None, str(error))
        return self.cache[key]

    def extract(self, mask_img):
        mask = np.asarray(mask_img.get_fdata()) > .5
        if mask.ndim != 3:
            raise ValueError('tSNR QC requires a 3D final ROI mask')
        size = int(mask.sum())
        rows = []
        for unit in self.units:
            entities, paths = unit['entities'], unit['files']
            row = {'session': entities.get('ses') or '', 'task': entities.get('task') or '',
                   'run': entities.get('run') or '', 'tsnr_file': ';'.join(paths),
                   'tsnr_mean': float('nan'), 'tsnr_median': float('nan'),
                   'tsnr_std': float('nan'), 'tsnr_n_voxels': 0, 'tsnr_coverage': 0., 'tsnr_error': ''}
            metadata = self.metadata[paths[0]] if len(paths) == 1 else {}
            row.update(tsnr_sidecar=str(tsnr_sidecar(paths[0])) if len(paths) == 1 else '',
                       tsnr_space=tsnr_space(paths[0], metadata) if len(paths) == 1 else '',
                       tsnr_registered=bool(metadata.get('ReferenceGrid')),
                       tsnr_reference_image=metadata.get('ReferenceImage', ''),
                       tsnr_source_template=metadata.get('SourceTemplate', ''),
                       tsnr_sources=json.dumps(metadata.get('Sources', [])))
            if size == 0:
                row['tsnr_status'] = 'empty_roi'
            elif not paths:
                row.update(tsnr_status='unresolved' if unit.get('error') else 'missing', tsnr_error=unit.get('error', ''))
            elif len(paths) != 1:
                row.update(tsnr_status='ambiguous', tsnr_error='Multiple maps match the same acquisition')
            else:
                values, valid, error = self._load(paths[0], mask_img.affine, mask.shape, unit)
                if error:
                    row.update(tsnr_status='invalid', tsnr_error=error)
                else:
                    selected = values[mask & valid]
                    count = int(selected.size)
                    row.update(tsnr_n_voxels=count, tsnr_coverage=count / size,
                               tsnr_status=('ok' if count == size else 'partial') if count else 'no_overlap')
                    if count:
                        row.update(tsnr_mean=float(selected.mean()), tsnr_median=float(np.median(selected)),
                                   tsnr_std=float(selected.std()))
            rows.append(row)
        usable = [row for row in rows if row['tsnr_n_voxels']]
        status = ('empty_roi' if size == 0 else 'missing' if not rows or all(row['tsnr_status']=='missing' for row in rows)
                  else 'ok' if all(row['tsnr_status']=='ok' for row in rows)
                  else 'partial' if usable else 'unavailable')
        metrics = {'tsnr_mean': float(np.mean([row['tsnr_mean'] for row in usable])) if usable else float('nan'),
                   'tsnr_n_maps': len(usable), 'tsnr_n_maps_expected': len(rows),
                   'tsnr_n_maps_available': sum(len(unit['files']) for unit in self.units),
                   'tsnr_voxel_coverage_mean': float(np.mean([row['tsnr_coverage'] for row in rows])) if rows else float('nan'),
                   'tsnr_status': status, 'tsnr_aggregation': 'equal_run_roi_mean'}
        logger.info('tSNR QC: status=%s | maps=%d/%d | mean=%s',
                    status, len(usable), len(rows), metrics['tsnr_mean'])
        return metrics, rows
