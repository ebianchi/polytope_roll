# File utilities

import os.path as op
import os


ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))
TEMP_DIR = f'{ROOT_DIR}/tmp'
OUT_DIR = f'{ROOT_DIR}/out'
