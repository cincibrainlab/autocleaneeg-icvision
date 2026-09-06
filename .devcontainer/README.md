# ICVision dev container

This container is for local development and offline validation. It installs the
project in editable mode with the `dev` and `test` extras, which provides
`pytest`, `mne`, `matplotlib`, `eeglabio`, and the rest of the science/test
stack that is missing from the host environment.

Open the repo in the dev container, then run:

```bash
python -m pytest tests/test_strict_api_safety.py
python experiments/runners/run_accuracy_sweep.py
PYTHONPYCACHEPREFIX=/tmp/icvision_pycache python -m py_compile \
  experiments/runners/run_accuracy_sweep.py \
  experiments/runners/run_screen.py \
  experiments/runners/build_accuracy_screen_manifest.py
```

For a local live run, mount or copy Grace's `IC_Visual_AI` directory into the
container and point the runner at it:

```bash
export ICVISION_GRACE_BASE_DIR=/path/to/IC_Visual_AI
```

Do not run the live ClinCog launch from this container until the non-code launch
gates are cleared: cblprod Grace-source provenance verification and PI
confirmation for ClinCog data transfer/IRB/retention terms.
