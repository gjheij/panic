import numpy as np

# helpers

def _gaussian_smooth3d(vol, sigma_vox):
    """Lightweight spatial smoothing; uses SciPy if available, else no-op."""
    if sigma_vox is None or sigma_vox <= 0:
        return vol
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(vol, sigma=sigma_vox, mode="reflect")
    except Exception:
        # SciPy not available; fall back without smoothing
        return vol

def _band_limited_tc(T, rng, low=0.005, high=0.08):
    """
    Make a single band-limited timecourse via FFT shaping (units are cycles/sample).
    low/high are fractions of Nyquist (0..0.5). Works without knowing TR.
    """
    x = rng.standard_normal(T)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(T)  # 0..0.5
    mask = (freqs >= low) & (freqs <= high)
    X *= mask.astype(float)
    y = np.fft.irfft(X, n=T)
    y -= y.mean()
    s = y.std(ddof=1) + 1e-8
    return y / s

def _find_trial_centers(design, n_trials, min_separation=3):
    """
    Heuristic peak picker from a convolved design (no access to raw events).
    Returns indices of the top n_trials separated by min_separation samples.
    """
    d = np.asarray(design, float)
    # Suppress edges to avoid boundary picks
    pad = max(2, min_separation)
    d[:pad] = 0; d[-pad:] = 0
    # Simple local maxima
    peaks = []
    for t in range(1, len(d)-1):
        if d[t] > d[t-1] and d[t] >= d[t+1]:
            peaks.append((d[t], t))
    peaks.sort(reverse=True)                    # by height
    centers = []
    for _, idx in peaks:
        if all(abs(idx - c) >= min_separation for c in centers):
            centers.append(idx)
            if len(centers) == n_trials:
                break
    centers.sort()
    return np.array(centers, dtype=int)

def _trial_windows(T, centers, half_width=4):
    """
    Build a (T, n_trials) matrix with smooth bell windows around each center.
    Uses a raised cosine for simplicity.
    """
    tt = np.arange(T)[:, None]
    cc = centers[None, :]
    w = np.clip(1 - ((tt - cc) / float(half_width))**2, 0, 1)  # parabolic
    return w

# main factory

def make_harder_noise(
    shape,
    T,
    rng,
    conv_face,
    conv_house,
    face_mask,
    house_mask,
    base_sd=0.3,          # base voxelwise SD floor
    k_snr=0.25,           # scales noise vs. signal per voxel (higher -> more noise)
    # temporal AR(1)
    ar1_mean=0.5,
    ar1_sd=0.15,
    # spatial correlation (in voxels)
    spatial_sigma=1.2,
    spatial_weight=0.4,
    # global components
    drift_strength=0.6,   # very low-frequency
    physio_strength=0.25, # mid-frequency
    global_strength=0.6,  # how much global couples into voxels
    # heteroscedasticity & spikes
    hetero_chunks=4,
    hetero_logsd=0.5,
    spike_prob=0.02,
    spike_scale=2.0,
    # trial-timed nuisance (hard mode)
    trial_jitter_strength=0.6,   # adds trial-timed nuisance within masks
    anti_contrast_prob=0.25,     # on some trials, oppose the true contrast
):
    """
    Returns: noise (X,Y,Z,T), sigma_vol (X,Y,Z), and a dict of components.
    """
    X, Y, Z = shape
    shape4 = shape + (T,)

    # per-voxel target SD tied to the signal's variability (like your code)
    # we'll compute sig_sd externally from your 'signal'
    # placeholder: function expects you to pass 'sig_sd' later or compute here
    def _compute_sigma_vol(signal):
        sig_sd = signal.std(axis=-1, ddof=1)
        sigma_vol = base_sd + k_snr * sig_sd
        return np.asarray(sigma_vol)

    # 1) AR(1) colored baseline noise (voxelwise rho)
    rho = np.clip(rng.normal(ar1_mean, ar1_sd, size=shape), 0.0, 0.98)
    eps = rng.standard_normal(shape4)
    ar1 = np.zeros_like(eps)
    ar1[..., 0] = eps[..., 0]
    for t in range(1, T):
        ar1[..., t] = rho * ar1[..., t-1] + eps[..., t]

    # 2) Spatially correlated field
    spatial = np.empty_like(ar1)
    for t in range(T):
        spatial[..., t] = _gaussian_smooth3d(rng.standard_normal(shape), spatial_sigma)
    # Mix with white to control effective strength
    spatial = spatial_weight * spatial + (1 - spatial_weight) * rng.standard_normal(shape4)

    # 3) Global signal: low-freq drift + band-limited 'physio'
    drift_tc  = _band_limited_tc(T, rng, low=0.0,   high=0.02) * drift_strength
    physio_tc = _band_limited_tc(T, rng, low=0.03,  high=0.15) * physio_strength
    g = (drift_tc + physio_tc)  # shape (T,)

    # Spatial weights for global coupling: smooth positive field
    gw = _gaussian_smooth3d(np.abs(rng.standard_normal(shape)), sigma_vox=1.5)
    gw = gw / (gw.std(ddof=1) + 1e-8) * global_strength
    global_component = gw[..., None] * g[None, None, None, :]

    # 4) Heteroscedasticity over time (piecewise constant SD multipliers)
    chunks = max(1, int(hetero_chunks))
    edges = np.linspace(0, T, chunks + 1, dtype=int)
    multipliers = np.ones(T, float)
    for i in range(chunks):
        m = np.exp(rng.normal(0, hetero_logsd))  # lognormal > 0
        multipliers[edges[i]:edges[i+1]] = m

    # 5) Sparse motion-like spikes (timepoint-specific)
    spikes = np.zeros(T, float)
    spike_times = np.where(rng.random(T) < spike_prob)[0]
    if spike_times.size:
        # spatial pattern with edges: difference of two blurred fields
        p1 = _gaussian_smooth3d(rng.standard_normal(shape), 1.0)
        p2 = _gaussian_smooth3d(rng.standard_normal(shape), 2.0)
        motion_pattern = (p1 - p2)
        motion_pattern /= (np.std(motion_pattern) + 1e-8)
        spikes[spike_times] = rng.lognormal(mean=np.log(spike_scale), sigma=0.3, size=spike_times.size)
        spike_component = motion_pattern[..., None] * spikes[None, None, None, :]
    else:
        spike_component = 0.0

    # 6) Trial-timed nuisance to specifically hurt decoding
    # Detect 14 trial centers from the total convolved design
    total_conv = np.asarray(conv_face) + np.asarray(conv_house)  # (T,)
    centers = _find_trial_centers(total_conv, n_trials=14, min_separation=3)
    W = _trial_windows(T, centers, half_width=4)                 # (T, 14)

    # Build a nuisance pattern that lives IN the same voxels as the real signal
    # so it competes directly with the contrast.
    # Base spatial pattern: face_mask (+1) vs house_mask (-1)
    contrast_map = (face_mask.astype(float) - house_mask.astype(float))
    contrast_map /= (np.sqrt((contrast_map**2).sum()) + 1e-8)    # unit energy

    # Per-trial amplitudes: mostly same-direction jitter, with some anti-contrast flips
    amps = rng.normal(0, 1, size=W.shape[1])
    flips = (rng.random(W.shape[1]) < anti_contrast_prob).astype(float)
    amps = amps * (1 - flips) - np.abs(amps) * flips  # flipped trials oppose the true effect

    trial_tc = (W * amps[None, :]).sum(axis=1)  # (T,)
    trial_tc /= (np.std(trial_tc) + 1e-8)
    trial_component = trial_jitter_strength * contrast_map[..., None] * trial_tc[None, None, None, :]

    # combine components (pre-scaling)
    noise_raw = (
        ar1 +
        spatial +
        global_component +
        spike_component +
        trial_component
    )

    # Heteroscedastic scaling across time
    noise_raw *= multipliers[None, None, None, :]

    # Standardize per-voxel and leave final scaling to match target SD
    noise_std = noise_raw.std(axis=-1, ddof=1, keepdims=True) + 1e-8
    noise_unit = noise_raw / noise_std

    return noise_unit, _compute_sigma_vol, {
        "rho_mean": float(rho.mean()),
        "n_spikes": int(len(spike_times)),
        "trial_centers": centers.tolist(),
    }
