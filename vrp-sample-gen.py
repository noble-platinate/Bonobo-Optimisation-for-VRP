import sys
import random
import math

nodescount = int(sys.argv[1])
maxcap = float(sys.argv[2])
minX = float(sys.argv[3])
maxX = float(sys.argv[4])
minY = float(sys.argv[5])
maxY = float(sys.argv[6])

print('params:')
print(f'  capacity {maxcap:.3f}')
print('nodes:')
for i in range(nodescount):
    demand = random.uniform(0.0, maxcap)
    x = random.uniform(minX, maxX)
    y = random.uniform(minY, maxY)
    # Calculate the number of leading zeros needed for the node labels
    label_width = math.ceil(math.log10(nodescount + 1))
    label = f'node{i+1:0{int(label_width)}d}'
    # Print node information with formatted output
    print(f'  {label}\t\t{demand:.3f}\t\t{x:.3f}\t\t{y:.3f}')
