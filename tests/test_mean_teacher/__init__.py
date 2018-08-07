import os
import sys


curr_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.join(curr_directory, '..', '..', '..')

sys.path.insert(0, project_directory)
