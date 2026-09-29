import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
  PROJECT_ROOT = _PROJECT_ROOT
  SFLM_ROOT = Path(os.getenv(
    'SFLM_ROOT', str(_PROJECT_ROOT.parent / 's-flm'))).resolve()
  RESULTS_DIR = Path(os.getenv(
    'SFMLM_RESULTS_DIR',
    str(_PROJECT_ROOT / 'sfmlm-experiments' / 'e1' / 'results'))).resolve()
