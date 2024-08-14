import sys
import random
import math

vrp = {}

# First reading the VRP from the input

def readinput():
    try:
        line = input().strip()
        while line == '' or line.startswith('#'):
            line = input().strip()
        return line
    except EOFError:
        return None

line = readinput()
if line is None:
    print('Empty input!', file=sys.stderr)
    exit(1)

if line.lower() != 'params:':
    print('Invalid input: it must be the VRP initial params at first!', file=sys.stderr)
    exit(1)

line = readinput()
if line is None:
    print('Invalid input: missing VRP initial params and nodes!', file=sys.stderr)
    exit(1)
while line.lower() != 'nodes:':
    inputs = line.split()
    if len(inputs) < 2:
        print('Invalid input: too few arguments for a param!', file=sys.stderr)
        exit(1)
    if inputs[0].lower() == 'capacity':
        vrp['capacity'] = float(inputs[1])
        # Validating positive non-zero capacity
        if vrp['capacity'] <= 0:
            print('Invalid input: capacity must be neither negative nor zero!', file=sys.stderr)
            exit(1)
    else:
        print('Invalid input: invalid VRP initial param!', file=sys.stderr)
        exit(1)
    line = readinput()
    if line is None:
        print('Invalid input: missing nodes!', file=sys.stderr)
        exit(1)

if not set(vrp).issuperset({'capacity'}):
    print('Invalid input: missing some required VRP initial params!', file=sys.stderr)
    exit(1)

line = readinput()
vrp['nodes'] = [{'label': 'depot', 'demand': 0, 'posX': 0, 'posY': 0}]
while line is not None:
    inputs = line.split()
    if len(inputs) < 4:
        print('Invalid input: too few arguments for a node!', file=sys.stderr)
        exit(1)
    node = {'label': inputs[0], 'demand': float(inputs[1]), 'posX': float(inputs[2]), 'posY': float(inputs[3])}
    # Validating demand neither negative nor zero
    if node['demand'] <= 0:
        print(f'Invalid input: the demand of the node {node["label"]} is negative or zero!', file=sys.stderr)
        exit(1)
    # Validating demand not greater than capacity
    if node['demand'] > vrp['capacity']:
        print(f'Invalid input: the demand of the node {node["label"]} is greater than the vehicle capacity!', file=sys.stderr)
        exit(1)
    vrp['nodes'].append(node)
    line = readinput()

# Validating no such nodes
if len(vrp['nodes']) == 0:
    print('Invalid input: no such nodes!', file=sys.stderr)
    exit(1)

# After inputting and validating it, now computing the algorithm

def distance(n1, n2):
    dx = n2['posX'] - n1['posX']
    dy = n2['posY'] - n1['posY']
    return math.sqrt(dx * dx + dy * dy)

def fitness(p):
    # The first distance is from depot to the first node of the first route
    s = distance(vrp['nodes'][0], vrp['nodes'][p[0]])
    # Then calculating the distances between the nodes
    for i in range(len(p) - 1):
        prev = vrp['nodes'][p[i]]
        next = vrp['nodes'][p[i + 1]]
        s += distance(prev, next)
    # The last distance is from the last node of the last route to the depot
    s += distance(vrp['nodes'][p[-1]], vrp['nodes'][0])
    return s

def adjust(p):
    # Adjust repeated
    repeated = True
    while repeated:
        repeated = False
        for i1 in range(len(p)):
            for i2 in range(i1):
                if p[i1] == p[i2]:
                    have_all = True
                    for node_id in range(len(vrp['nodes'])):
                        if node_id not in p:
                            p[i1] = node_id
                            have_all = False
                            break
                    if have_all:
                        del p[i1]
                    repeated = True
                if repeated: break
            if repeated: break
    # Adjust capacity exceed
    i = 0
    s = 0.0
    cap = vrp['capacity']
    while i < len(p):
        s += vrp['nodes'][p[i]]['demand']
        if s > cap:
            p.insert(i, 0)
            s = 0.0
        i += 1
    i = len(p) - 2
    # Adjust two consecutive depots
    while i >= 0:
        if p[i] == 0 and p[i + 1] == 0:
            del p[i]
        i -= 1

popsize = int(sys.argv[1])
iterations = int(sys.argv[2])

pop = []

# Generating random initial population
for i in range(popsize):
    p = list(range(1, len(vrp['nodes'])))
    random.shuffle(p)
    pop.append(p)
for p in pop:
    adjust(p)

# Running the genetic algorithm
for i in range(iterations):
    next_pop = []
    # Each one of this iteration will generate two descendants individuals. Therefore, to guarantee same population size, this will iterate half population size times
    for j in range(int(len(pop) / 2)):
        # Selecting randomly 4 individuals to select 2 parents by a binary tournament
        parent_ids = set()
        while len(parent_ids) < 4:
            parent_ids |= {random.randint(0, len(pop) - 1)}
        parent_ids = list(parent_ids)
        # Selecting 2 parents with the binary tournament
        parent1 = pop[parent_ids[0]] if fitness(pop[parent_ids[0]]) < fitness(pop[parent_ids[1]]) else pop[parent_ids[1]]
        parent2 = pop[parent_ids[2]] if fitness(pop[parent_ids[2]]) < fitness(pop[parent_ids[3]]) else pop[parent_ids[3]]
        # Selecting two random cutting points for crossover, with the same points (indexes) for both parents, based on the shortest parent
        cut_idx1, cut_idx2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
        cut_idx1, cut_idx2 = min(cut_idx1, cut_idx2), max(cut_idx1, cut_idx2)
        # Doing crossover and generating two children
        child1 = parent1[:cut_idx1] + parent2[cut_idx1:cut_idx2] + parent1[cut_idx2:]
        child2 = parent2[:cut_idx1] + parent1[cut_idx1:cut_idx2] + parent2[cut_idx2:]
        next_pop += [child1, child2]
    # Doing mutation: swapping two positions in one of the individuals, with 1:15 probability
    if random.randint(1, 15) == 1:
        to_mutate = next_pop[random.randint(0, len(next_pop) - 1)]
        i1 = random.randint(0, len(to_mutate) - 1)
        i2 = random.randint(0, len(to_mutate) - 1)
        to_mutate[i1], to_mutate[i2] = to_mutate[i2], to_mutate[i1]
    # Adjusting individuals
    for p in next_pop:
        adjust(p)
    # Updating population generation
    pop = next_pop

# Selecting the best individual, which is the final solution
better = None
best_fitness = float('inf')
for p in pop:
    f = fitness(p)
    if f < best_fitness:
        best_fitness = f
        better = p

# Printing the solution
print("0")
for node_idx in better:
    x=vrp['nodes'][node_idx]['label']
    if(x=="depot"):
        print("0")
    else: 
        print(x[4:])
print("0")
print(f"{best_fitness:.6f}", end="")