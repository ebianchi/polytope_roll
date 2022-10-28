# File utilities

import os.path as op
import os


ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))
TEMP_DIR = f'{ROOT_DIR}/tmp'
OUT_DIR = f'{ROOT_DIR}/out'


def assure_created(directory: str) -> str:
    """A function from the DAIR PLL repository:  Wrapper to put around directory
    paths which ensure their existence.

    Args:
        directory: Path of directory that may not exist.

    Returns:
        ``directory``, Which is ensured to exist by recursive mkdir.
    """
    directory = op.abspath(directory)
    if not op.exists(directory):
        assure_created(op.dirname(directory))
        os.mkdir(directory)
    return directory


# Make sure the temp and out directories exist, if not already.
assure_created(TEMP_DIR)
assure_created(OUT_DIR)