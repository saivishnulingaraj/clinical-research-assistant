"""Convenience entrypoints exposed via ``python -m cra`` / console scripts."""

from __future__ import annotations


def eval_entrypoint() -> None:
    """Console-script shim for running evals.

    Implemented in the evals milestone; kept importable here so packaging
    metadata resolves from the start.
    """
    from cra.evals.runner import main

    main()


if __name__ == "__main__":  # pragma: no cover
    eval_entrypoint()
