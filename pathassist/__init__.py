"""Digital pathology assistant — triage, explainability, reporting, and learning from corrections.

The package is organised as a linear, human-in-the-loop pipeline:

    whole-slide image
        -> tiling            (tiling.py)
        -> quality control    (qc.py)
        -> region scoring     (scoring.py — dummy, single model, or ensemble)
        -> case triage        (triage.py)
        -> severity grading   (grading.py)
        -> explainability     (explain.py)
        -> report drafting    (report.py)
        -> human approval     (audit.py)

Every stage records what it did so the run is auditable end to end. No stage
ever finalises a diagnosis; the output is always a draft awaiting a pathologist.
"""

__version__ = "0.2.0"
