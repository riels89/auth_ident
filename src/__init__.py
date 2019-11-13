TRAIN_LEN = 1000000 # 1e6
VAL_LEN = 50000
TEST_LEN = 50000
DATA_SIZE = TRAIN_LEN + VAL_LEN + TEST_LEN

import sys
if sys.platform is "win32":
    SL = "\\"
else:
    SL = "/"
