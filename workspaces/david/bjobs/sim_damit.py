import sys
from hac.damit import DamitDataset

idx = int(sys.argv[1]) - 1

ds = DamitDataset()
ds[idx]
print("Success")