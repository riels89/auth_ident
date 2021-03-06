import sys
import os
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocessing.by_line_dataset import by_line_dataset
dataset = by_line_dataset(max_lines=50, max_line_length=120, batch_size=64, binary_encoding=False)
train, val = dataset.get_dataset()

for batch in train:
	print(str(BytesInUse()))
