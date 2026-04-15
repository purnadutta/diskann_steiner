"""HiddenBridge package."""


def run_diskann_steiner_experiment(*args, **kwargs):
    from hiddenbridge.experiment import run_diskann_steiner_experiment as _run

    return _run(*args, **kwargs)


__all__ = ["run_diskann_steiner_experiment"]
