"""Local agent-guided ionic-liquid design utilities."""


def run_design_agent(*args, **kwargs):
    from .design_agent import run_design_agent as _run_design_agent

    return _run_design_agent(*args, **kwargs)


__all__ = ["run_design_agent"]
