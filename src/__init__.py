TRAIN_LEN = 10000#1000000 # 1e6
VAL_LEN = 500#50000
TEST_LEN = 50000
DATA_SIZE = TRAIN_LEN + VAL_LEN + TEST_LEN

import sys
if sys.platform == "win32":
    SL = "\\"
else:
    SL = "/"
