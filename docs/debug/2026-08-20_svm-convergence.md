# Searchlight end-of-run stall: debugging report

- Debug report: SVM convergence issue
- Date: 2026-08-20
- Branch: debug/searchlight-tail-stall
- Status: resolved

## Summary

A reproducible end-of-run stall affected the Python/joblib fMRI
searchlight decoder when processing approximately 222,000 voxel centers
with 10 label permutations per center. Runs progressed normally at
roughly 70--90 centers/s and then appeared to hang near completion. The
same behavior occurred with both joblib's `loky` and `multiprocessing`
backends.

Instrumentation ultimately showed that this was **not a joblib
result-queue deadlock, file-descriptor leak, memmap accumulation,
pathological final center, or a general Slurm/node failure**. One worker
remained alive and CPU-running inside scikit-learn/libsvm while fitting
an `SVC` for a particular permutation. The problematic task was:

-   center index: `58116`
-   stage: `BEFORE_PERM_9`
-   production worker: PID `1354848`
-   observed stale duration: \>2300 s
-   process state: `R (running)`
-   worker remained CPU-active rather than sleeping in kernel I/O
-   file descriptors and memory-map counts remained stable

A fresh-process replay of exactly center 58116 and permutation 9
completed normally in milliseconds. Its three outer-fold SVC fits
converged in only 427, 507, and 365 iterations. Thus the input itself
was not intrinsically pathological.

The practical fix was to configure a **finite `SVC.max_iter`**. A full
searchlight with a finite iteration limit completed successfully. The
limit acts as a safety fuse preventing an anomalous libsvm fit in a
long-lived worker from monopolizing one worker indefinitely and
preventing the final `Parallel` result from completing.

------------------------------------------------------------------------

## Original failure

Typical searchlight configuration:

``` text
X = (135, 221989)
centers = 221989
radius = 5–6 mm
permutations = 10
n_jobs = 10–12
batch_size = 1
backend = loky (later also multiprocessing)
```

Normal throughput was approximately 70--90 centers/s. Runs commonly
reached:

``` text
215000/221989
220000/221989
```

with an ETA below one minute, but then remained alive for tens of
minutes or longer.

Because progress counted **completed tasks**, not center indices in
strict order, `220000/221989` did not establish which task was missing.

------------------------------------------------------------------------

## Hypotheses initially considered

The evidence initially supported a broad
cumulative-resource/process-lifetime problem. Candidate causes included:

-   joblib/loky scheduling or result IPC
-   multiprocessing queues or semaphores
-   dead workers
-   parent waiting indefinitely for a missing result
-   file-descriptor leakage
-   repeated `numpy.memmap` mappings
-   joblib `resource_tracker` behavior
-   Lustre/filesystem blocking
-   accumulated worker memory/state
-   BLAS/OpenMP oversubscription
-   tqdm/joblib callback interaction
-   a pathological searchlight center
-   a pathological SVC optimization
-   long-lived libsvm/scikit-learn process state

Several of these were progressively ruled out.

------------------------------------------------------------------------

## Tests and attempted fixes

### Serial replay of the tail

The final approximately 2,489 centers were run serially.

All centers, including the final centers, completed normally at about
0.12 s per center. Output maps were saved and the process exited
normally.

**Conclusion:** there was no inherently bad final center.

### Parallel replay of the tail

The same final approximately 2,489 centers were run in parallel.

They completed normally at approximately 70 centers/s.

**Conclusion:** the tail itself was also valid under parallel execution.

### `batch_size=1`

The full searchlight already used:

``` python
batch_size=1
```

Therefore the stall was not a large final joblib batch containing one
slow task.

### Thread limitation

`OPENBLAS_NUM_THREADS=1` was already configured. Later
version/threadpool information also showed one OpenBLAS thread and one
OpenMP thread.

**Conclusion:** ordinary numerical thread oversubscription was not the
explanation.

### `return_as="generator_unordered"`

Joblib was tested using an explicitly consumed unordered result
generator.

The full run still stalled.

An earlier apparent success caused by creating but not consuming the
generator was discarded.

**Conclusion:** result ordering was not the underlying problem.

### In-process chunking

The workload was divided into approximately 20,000-center chunks with
separate `Parallel()` calls in the same Python process.

The stall still occurred and could occur much earlier than center
220,000.

**Conclusion:** the problem was associated with sustained process/worker
execution rather than the numerical index at the end of the mask. Merely
recreating `Parallel` objects inside the same Python process was
insufficient.

### `loky` versus `multiprocessing`

The original implementation used `loky`.

The worker was moved to module scope so that the standard
multiprocessing backend could pickle it. A complete multiprocessing run
then reproduced the same near-end stall.

**Conclusion:** the problem was not specific to loky's executor
implementation.

### Resource monitoring

Per-worker diagnostics periodically recorded:

-   open file-descriptor count
-   `/proc/<pid>/maps` count
-   X memmap mapping count
-   RSS
-   virtual memory size
-   thread count

For a representative worker over thousands of tasks, values remained
essentially stable:

``` text
fd_count:    ~15
map_count:   ~1885–1888
RSS:         ~276–289 MB
threads:     1
```

There was no monotonic file-descriptor or mapping explosion.

**Conclusion:** repeated searchlight execution did not show evidence of
a simple FD/mmap leak sufficient to explain the stall.

------------------------------------------------------------------------

## Critical instrumentation

A separate `monitor.py` module was introduced.

Workers wrote:

1.  an atomic heartbeat file containing PID, center index, stage, and
    timestamp;
2.  a per-worker completion journal containing completed center indices;
3.  periodic process-resource snapshots.

The parent maintained an exact completion bitmap and periodically
reported outstanding center indices.

Worker stages included locations such as:

``` text
BEFORE_LOAD
AFTER_LOAD
BEFORE_OBS
AFTER_OBS
BEFORE_PERM_0
...
BEFORE_PERM_9
AFTER_PERM_9
DONE
```

For stale workers, the monitor inspected `/proc/<pid>` and optionally
sent `SIGUSR1` to trigger a `faulthandler` stack dump.

This changed the problem from "the searchlight hangs around 99%" to
identification of the exact missing task.

------------------------------------------------------------------------

## Identification of the stuck task

During a failing full run, the monitor eventually reported:

``` text
completed=221988/221989
outstanding=1
first=[58116]
```

All other workers were `DONE`. One worker remained:

``` text
pid=1354848
ix=58116
stage=BEFORE_PERM_9
age_s=>2300
```

Earlier stale-worker diagnostics showed:

``` text
State: R (running)
wchan: 0
Threads: 1
fd_count: 14
map_count: 1888
```

CPU accounting (`utime_ticks`) continued increasing.

This was decisive.

**Interpretation:** the worker was alive and actively consuming CPU. It
was not blocked on Lustre, a pipe, a result queue, a semaphore, or
another obvious kernel wait.

The parent was waiting because one legitimate worker computation never
returned.

------------------------------------------------------------------------

## Stack trace

`SIGUSR1`/`faulthandler` showed that the worker was executing within the
SVM fitting path. Subsequent instrumentation focused on scikit-learn's
`SVC`/libsvm solver.

The environment was:

``` text
Python:       3.11.15
scikit-learn: 1.9.0
NumPy:        1.26.4
SciPy:        1.17.1
joblib:       1.5.3
OpenBLAS:     0.3.33
```

BLAS and OpenMP were each constrained to one thread.

------------------------------------------------------------------------

## Exact center replay

A pytest replay was built for subject `sub-017`, using the same
whole-brain searchlight mask and reconstructing center 58116.

The center was:

``` text
center index = 58116
ijk          = (32, 91, 39)
features     = 177
samples      = 135
permutations = 10
```

The exact permutation-9 seed was:

``` text
2114766328
```

The replay executed the observed score and permutations 0 through 9 in
the same fresh process.

Everything completed normally.

Permutation 9 took approximately 0.012 s in total.

------------------------------------------------------------------------

## SVC iteration-count instrumentation

After each successful `clf.fit`, the fitted final estimator was
inspected:

``` python
fitted = clf.best_estimator_ if isinstance(clf, BaseSearchCV) else clf

estimator = (
    fitted.named_steps.get("clf")
    if hasattr(fitted, "named_steps")
    else fitted
)

n_iter = getattr(estimator, "n_iter_", None)
fit_status = getattr(estimator, "fit_status_", None)
max_iter = getattr(estimator, "max_iter", None)
```

For the exact center/permutation that had run for tens of minutes in
production, the fresh replay produced:

``` text
PERM 9
fold 0: n_iter = 427
fold 1: n_iter = 507
fold 2: n_iter = 365
```

with:

``` text
fit_status = 0
max_iter   = 100000
```

Across the replay, normal iteration counts were generally hundreds to
roughly 1,300.

**Conclusion:** center 58116/permutation 9 is not intrinsically a
difficult SVC optimization. The production runaway depended on process
history/state.

------------------------------------------------------------------------

## Final practical solution

A finite `max_iter` was added to the `SVC` configuration.

A subsequent full searchlight completed:

``` text
221902/221989
...
Saving output maps
Done
Decoding sub-017 complete
```

No worker was allowed to remain indefinitely inside an anomalous libsvm
optimization.

The finite iteration count should therefore be viewed as a **solver
safety fuse**.

It does not prove the precise internal libsvm mechanism that caused the
long-lived worker to enter the pathological fit, but it prevents that
failure mode from stalling the entire searchlight.

The exact problematic center requires only a few hundred iterations in a
fresh process, so a cap such as 10,000 already leaves substantial
headroom for normal searchlight fits.

------------------------------------------------------------------------

## Separate convergence issue in ROI/GridSearchCV analyses

Finite `max_iter` also exposed ordinary convergence warnings elsewhere.

In ROI analyses with `GridSearchCV`, convergence warnings could occur
during permutations inside an inner grid-search candidate/fold. For
example, an SVC reached even `max_iter=100000`.

That path is distinct from the searchlight failure:

``` text
permutation
  -> outer CV
    -> GridSearchCV
      -> inner candidate/fold
        -> SVC.fit
```

The searchlight effective configuration disables feature selection and
grid search and uses a locked classifier parameter.

Therefore:

-   the searchlight runaway should not be interpreted as evidence that
    center 58116 genuinely needs \>10,000 iterations;
-   occasional grid-search convergence failures during permuted ROI
    analyses should be investigated separately by
    candidate/fold/hyperparameter.

------------------------------------------------------------------------

## Current interpretation

The evidence supports the following causal chain:

``` text
long-lived searchlight worker
        |
        v
one SVC/libsvm fit enters pathological solver behavior
        |
        v
worker remains CPU-running inside fit
        |
        v
all other centers eventually complete
        |
        v
joblib parent legitimately waits for the one missing result
        |
        v
searchlight appears to "hang" at ~99–100%
```

The evidence argues against:

``` text
pathological final center
joblib batch-size artifact
result ordering
loky-specific bug
simple multiprocessing queue deadlock
dead worker
filesystem wait
OpenBLAS oversubscription
FD leak
mmap-count leak
parent progress callback as original cause
```

The exact low-level reason libsvm behaves pathologically only after
sustained worker execution remains unresolved. Worker recycling remains
a potentially useful follow-up experiment, but it is not required for
the current safety fix.

------------------------------------------------------------------------

## Monitoring retained for future diagnosis

The monitoring infrastructure remains useful and should be retained at
low log verbosity.

Recommended levels:

``` text
INFO
    Searchlight progress, throughput, elapsed time, ETA
    lifecycle events

DEBUG
    SEARCHLIGHT_WATCHDOG
    SEARCHLIGHT_WORKER_RESOURCE
    normal SVC_ITER telemetry

WARNING
    SEARCHLIGHT_STUCK_WORKER
    SVC max-iteration/convergence events

ERROR
    actual computation failures
```

Example human-facing progress message:

``` text
Searchlight progress: 220000/221989 (99.1%) |
72.14 centers/s | elapsed 50.8 min | ETA 0.5 min
```

The monitor should continue recording exact outstanding indices and
worker heartbeats even when those snapshots are only emitted at `DEBUG`
level.

------------------------------------------------------------------------

## Monitor regression discovered during cleanup

While restoring ETA logging, a separate monitor-only deadlock was
introduced accidentally.

`mark_completed()` held `self._lock` while calling:

``` python
self.dump_snapshot(reason="progress")
```

`dump_snapshot()` calls `outstanding()`, which attempts to acquire the
same non-reentrant `threading.Lock`.

The resulting sequence was:

``` text
mark_completed()
  -> acquire self._lock
  -> reach 5000 completions
  -> log progress
  -> dump_snapshot()
  -> outstanding()
  -> acquire self._lock again
  -> deadlock
```

This explained a later test that stopped immediately after:

``` text
Searchlight progress: 5000/221989 ...
```

The fix was simply to release the lock before progress logging and
`dump_snapshot()`.

This regression is **not the original searchlight/libsvm stall** and
should be kept conceptually separate from it.

------------------------------------------------------------------------

## Key lessons

1.  Progress at 99% did not imply the final center was problematic;
    unordered task completion made exact outstanding-index tracking
    essential.
2.  Replaying only the tail could not reproduce a
    process-history-dependent failure.
3.  Worker heartbeats with explicit computation stages were much more
    informative than joblib-level progress alone.
4.  `/proc` state distinguished a CPU-running numerical solver from
    IPC/filesystem blocking.
5.  Stable FD/map/RSS measurements substantially weakened the
    resource-leak hypothesis.
6.  Exact deterministic replay demonstrated that the offending
    center/permutation was numerically normal in a fresh process.
7.  A finite solver iteration cap is important defensive configuration
    for extremely large repeated-fit workloads even when ordinary fits
    converge far below that limit.
8.  Monitoring code itself must avoid holding locks while calling
    routines that reacquire those locks.

------------------------------------------------------------------------

## Final status

The original full-run stall has been isolated to a single CPU-running
SVC/libsvm fit in a long-lived worker. Setting a finite `SVC.max_iter`
allowed the full approximately 222,000-center searchlight to finish and
save its maps normally.

The monitor now provides enough information to immediately distinguish
any future recurrence among:

-   a genuinely slow numerical task,
-   a dead/stale worker,
-   parent/result IPC problems,
-   filesystem/resource problems,
-   and ordinary healthy progress.
