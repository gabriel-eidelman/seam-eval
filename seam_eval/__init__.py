"""
SeamEval — handoff-level failure attribution for multi-agent LLM systems.

Quick imports
-------------
    from seam_eval.taxonomy import HandoffFailureMode, SeamEvent, SeamTrace
    from seam_eval.adapters.autogen import AutoGenAdapter
    from seam_eval.callbacks.seam_trace import SeamTraceCallback
    from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator
    from seam_eval.benchmarks.seam_benchmark import SeamBenchmark
"""

from seam_eval.taxonomy import HandoffFailureMode, SeamEvent, SeamTrace
from seam_eval.callbacks.seam_trace import SeamTraceCallback
from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator
from seam_eval.benchmarks.seam_benchmark import SeamBenchmark

__all__ = [
    "HandoffFailureMode",
    "SeamEvent",
    "SeamTrace",
    "SeamTraceCallback",
    "HandoffEvaluator",
    "SeamBenchmark",
]

__version__ = "0.1.0"
