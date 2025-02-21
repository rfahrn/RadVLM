import os
import sys

DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR is None:
    raise EnvironmentError("The environment variable 'DATA_DIR' is not set.")
