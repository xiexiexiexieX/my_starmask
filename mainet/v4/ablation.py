"""Compatibility entry point for the isolated V4 ablation experiment.

New commands should use ``mainet/v4/ablation_experiment/run.py``. This file is
kept so existing imports and old command lines continue to work.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ablation_experiment.run import *
from ablation_experiment.run import main


if __name__ == '__main__':
    main()
