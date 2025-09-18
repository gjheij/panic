# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from bids import BIDSLayout
import nibabel as nib
import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.signal import fftconvolve
from scipy.stats import gamma
from pathlib import Path
from types import SimpleNamespace
from stglm.cli import run_stglm_from_inputs
from panic import noise
from lazyfmri import plotting, utils

shape: tuple[int, int, int] = (10, 10, 10)
n_timepoints: int = 100
amplitude: float = 5.0
repetition_time: float = 2.0
voxel_size: float = 2.0

# Some onsets with jitter
onsets = [
    10.257801082299062,
    28.048984370225746,
    43.323283486204126,
    56.34091764205242,
    69.47620661836409,
    77.49889380935444,
    86.13188781960628,
    96.8891562508195,
    106.3304476228412,
    114.43813702339415,
    131.24871513882664,
    146.2612368172774,
    156.64656043381945,
    168.05379551502983,
]

duration = 0.0

# set order
conditions = [
    "house",
    "house",
    "face",
    "face",
    "house",
    "face",
    "face",
    "house",
    "face",
    "face",
    "house",
    "house",
    "face",
    "house",
]

def spm_hrf(repetition_time: float, oversampling: float, time_length: float = 32.0) -> npt.NDArray[np.float64]:
    """
    Return values for HRF at temporal resolution repetition_time/oversampling.
    Canonical SPM double-gamma HRF.
    """
    dt = repetition_time / float(oversampling)
    time_vec = np.arange(0, time_length, dt)
    # parameters for the two gamma functions
    peak1 = gamma.pdf(time_vec, 6)  # peak at 6s
    peak2 = gamma.pdf(time_vec, 16) * 0.35  # undershoot
    hrf = peak1 - peak2
    hrf = hrf / np.sum(hrf)  # normalize area=1
    return hrf


def convolve_hrf_from_onsets(
    onsets: list[float], hrf: npt.NDArray[np.float64], total_time: float, time_step: float = 0.1
) -> npt.NDArray[np.float64]:
    """
    Generate an onset vector at high resolution and convolve with HRF.

    Parameters:
    - onsets: list of event onset times (in seconds)
    - hrf: high-resolution HRF (sampled at time_step)
    - total_time: total duration of scan (in seconds)
    - time_step: high-resolution sampling step (in seconds)

    Returns:
    - convolved signal, downsampled to TR resolution
    """
    n_hr = int(total_time / time_step)
    n_tr = int(total_time / repetition_time)
    highres_vector = np.zeros(n_hr)

    # Fill event onsets
    for onset in onsets:
        idx = int(onset / time_step)
        if idx < len(highres_vector):
            highres_vector[idx] = 1.0

    # Convolve
    full = fftconvolve(highres_vector, hrf)[:n_hr]

    # Downsample to TR resolution
    downsampled = full[:: int(repetition_time / time_step)]
    return downsampled[:n_tr]  # clip to match timepoints


def save_4d_nifti(data_4d: npt.NDArray[np.float64], out_path: Path) -> None:
    """
    Save a single 4D NIfTI file with given voxel size (mm) and TR (s).
    """
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(xyz="mm", t="sec")
    hdr["pixdim"][1:4] = voxel_size
    hdr["pixdim"][4] = repetition_time
    img = nib.Nifti1Image(data_4d, affine, header=hdr)
    nib.save(img, out_path)


def plot_example(data_4d: npt.NDArray[np.float64], out_path: Path=None, **kwargs) -> None:
    import matplotlib.pyplot as plt

    # Pick example voxels
    face_voxel = (8, 5, 5)  # in the "face" half (x >= 5)
    house_voxel = (2, 5, 5)  # in the "house" half (x < 5)

    # Extract timecourses
    tc_face = data_4d[face_voxel]
    tc_house = data_4d[house_voxel]

    # Plot
    fig, axs = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True
    )

    defaults = {
        "line_width": 2,
        "x_label": "Time (TR)",
        "y_label": "Signal intensity",
        "title": "example timeseries",
        "labels": ["Face", "House"],
        "add_hline": 0
    }

    for key, val in defaults.items():
        kwargs = utils.update_kwargs(
            kwargs,
            key,
            val
        )

    _ = plotting.LazyLine(
        [tc_face, tc_house],
        axs=axs,
        **kwargs
    )

    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)

def half_masks() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns two masks (house_mask, face_mask) splitting along x-axis.
    """
    house = np.zeros(shape, dtype=float)
    house[: shape[0] // 2, :, :] = 1.0
    face = np.zeros(shape, dtype=float)
    face[shape[0] // 2 :, :, :] = 1.0
    return house, face

    
def events_df() -> pd.DataFrame:
    """
    Given onsets array and conditions list,
    returns a pandas DataFrame with onset, duration, trial_type.
    """

    events = []
    for onset, cond in zip(onsets, conditions, strict=False):
        events.append(
            {
                "onset": float(onset),
                "duration": float(duration),
                "trial_type": cond
            }
        )
    return pd.DataFrame(events)

def condition_file(tmp_path_factory, events_df: pd.DataFrame) -> Path:
    tmp_path_factory.mkdir(exist_ok=True)

    condition_file = tmp_path_factory / "sub-01_ses-1_task-HousesFaces_run-1_events.tsv"
    events_df.to_csv(condition_file, sep="\t", index=False)
    return condition_file

def simulated_bold_file(
    tmp_path_factory,
    half_masks: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    events_df: pd.DataFrame,
    use_hrf_mismatch: bool=False,
    hrf_shift: float=1,
    plot_kws: dict={},
    **kwargs
) -> Path:
    """
    Create 4D data: Gaussian noise + HRF-convolved activations from events_df.
    Returns path to a NIfTI file.

    use_hrf_mismatch: False
        use a *different* HRF to SIMULATE than the one you'll FIT in the GLM (double gamma)
    base_sd: 0
        background noise everywhere (SD)
    k_snr: 0
        scales noise by local signal SD; higher => harder
    rho_ar1: 0
        temporal autocorrelation (0 = white)
    """

    rng = np.random.default_rng(0)
    tmp_path_factory.mkdir(exist_ok=True)

    # Generate 4D data with HRF convolution
    house_mask, face_mask = half_masks

    # high-resolution HRF
    time_step = 0.1
    hrf = spm_hrf(repetition_time=time_step, oversampling=1)
    hrf /= hrf.max()

    hrf_sim = hrf
    if use_hrf_mismatch:
        # delay + a bit wider
        shift = int(round(hrf_shift / time_step))
        hrf_sim = np.roll(hrf, shift)

    total_time = n_timepoints * repetition_time

    # split onsets by condition
    onsets_by_type = {
        "face": events_df.query("trial_type == 'face'")["onset"].tolist(),
        "house": events_df.query("trial_type == 'house'")["onset"].tolist(),
    }

    conv_face = convolve_hrf_from_onsets(
        onsets_by_type["face"],
        hrf_sim,
        total_time,
        time_step=time_step
    ) * amplitude

    conv_house = convolve_hrf_from_onsets(
        onsets_by_type["house"],
        hrf_sim, 
        total_time,
        time_step=time_step
    ) * amplitude

    conv_face  = np.asarray(conv_face)
    conv_house = np.asarray(conv_house)

    # Assemble 4D data
    signal = (
        face_mask[..., None]  * conv_face[None, None, None, :] +
        house_mask[..., None] * conv_house[None, None, None, :]
    )

    T = n_timepoints
    shape = (10, 10, 10)  # as you described

    noise_unit, sigma_fn, meta = noise.make_harder_noise(
        shape, T, rng,
        conv_face=conv_face,
        conv_house=conv_house,
        face_mask=face_mask,
        house_mask=house_mask,
        **kwargs
    )

    # Final per-voxel target SD based on YOUR constructed 'signal'
    sigma_vol = sigma_fn(signal)                            # (X,Y,Z)
    final_noise = noise_unit * sigma_vol[..., None]               # scale

    # Final data
    data = signal + final_noise

    for i in range(2):
        plot_example(
            data,
            tmp_path_factory / "example_timecourses.png",
            **plot_kws
        )

    # Save volumes
    bold_file = tmp_path_factory / "sub-01_ses-1_task-HousesFaces_run-1_bold.nii.gz"
    save_4d_nifti(data, bold_file)
    save_4d_nifti(house_mask, tmp_path_factory / "house.nii.gz")
    save_4d_nifti(face_mask, tmp_path_factory / "face.nii.gz")

    return bold_file, tmp_path_factory / "face.nii.gz"

def build_inputs(
    tmp_path,
    hrf="dgamma",
    lss_mode="cond",
    run=True,
    **kwargs
):

    event_df = events_df()
    masks = half_masks()
    _ = condition_file(tmp_path, event_df)
    _ = simulated_bold_file(
        tmp_path,
        masks,
        event_df,
        **kwargs
    )

    layout = BIDSLayout(tmp_path, validate=False)
    bold_file = layout.get(suffix="bold")
    ev_file = layout.get(suffix="events", return_type="file")
    evs = np.unique(event_df["trial_type"]).tolist()

    ddict = {
        "hrf": hrf,
        "mode": lss_mode,
        "confounds": None,
        "confounds_regex": None,
        "conditions": evs
    }

    feature = SimpleNamespace(**ddict)
    
    print(bold_file[0].path)
    runs = []
    runs.append({
        "bold": bold_file[0].path,
        "mask": masks[0],
        "confounds": None,
        "events": ev_file,
        "tags": bold_file[0].entities,
        "tr": repetition_time
    })

    if run:
        result = run_stglm_from_inputs(
            run_data_list=runs,
            feature=feature,
            workdir=tmp_path,
            confound_regexes=None,
            n_procs=1,
            derivatives=str(tmp_path / "derivatives")
        )

        return result
    else:
        return runs