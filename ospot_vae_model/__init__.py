import sys
from os import path

lib_path = path.abspath(path.dirname(__file__))
if lib_path not in sys.path:
    sys.path.insert(0,lib_path)
