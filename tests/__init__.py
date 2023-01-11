import os

## File intended for making it easier to refere to data files while testing

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data
_PATH_Models = os.path.join(_PROJECT_ROOT, "Src/models")  # root of data

## In the test scipts then, the data path can be imported by 
#  from tests import _PATH_DATA
