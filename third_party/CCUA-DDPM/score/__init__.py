"""Metric compatibility namespace owned by the CCUA runtime.

The upstream CCUA U-Net checkout keeps its released metric implementation in
``score_new``.  The runpack's shared evaluator imports the conventional
``score.*`` names, so these small shims keep metric execution inside the one
active vendor tree without introducing CBDM as a runtime dependency.
"""

