"""Digital pathology assistant - a decision-support scaffold.

The package is organised as a linear, human-in-the-loop pipeline:

    whole-slide image
        -> tiling            (tiling.py)
        -> region scoring     (scoring.py)
        -> case triage        (triage.py)
        -> explainability     (explain.py)
        -> report drafting    (report.py)
        -> human approval     (audit.py)

Every stage records what it did so the run is auditable end to end. No stage
ever finalises a diagnosis; the output is always a draft awaiting a pathologist.
"""

__version__ = "0.1.0"
