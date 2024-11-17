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
                    for node_id in range(1, len(vrp['nodes'])):
                        if node_id not in p:
                            p[i1] = node_id
                            have_all = False
                            break
                    if have_all:
                        del p[i1]
                    repeated = True
                    break
            if repeated:
                break
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

# Crossover Operators

def order_crossover_1(parent1, parent2):
    size = len(parent1)
    child = [None]*size

    # Select two crossover points at random
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    start, end = min(cx_point1, cx_point2), max(cx_point1, cx_point2)

    # Copy the segment from parent2 to child
    child[start:end+1] = parent2[start:end+1]

    # Fill the remaining positions with the order of elements from parent1
    pos = (end + 1) % size
    p1_pos = pos
    while None in child:
        if parent1[p1_pos] not in child:
            child[pos] = parent1[p1_pos]
            pos = (pos + 1) % size
        p1_pos = (p1_pos + 1) % size
    return child

def order_crossover_2(parent1, parent2):
    size = len(parent1)
    child = [None]*size

    # Randomly select a set of positions
    num_positions = random.randint(1, size)
    positions = random.sample(range(size), num_positions)
    positions.sort()

    # Copy the elements from parent2 at the selected positions to child
    for pos in positions:
        child[pos] = parent2[pos]

    # Fill the remaining positions with the order of elements from parent1
    p1_elements = [elem for elem in parent1 if elem not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = p1_elements.pop(0)
    return child

def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child = [None]*size
    cycles = []
    indices = list(range(size))
    while indices:
        idx = indices[0]
        cycle = []
        val = parent1[idx]
        while True:
            cycle.append(idx)
            idx = parent1.index(parent2[idx])
            if idx == cycle[0]:
                break
        cycles.append(cycle)
        indices = [i for i in indices if i not in cycle]
    # Copy elements from parent1 or parent2 based on cycle parity
    for i, cycle in enumerate(cycles):
        parent = parent1 if i % 2 == 0 else parent2
        for idx in cycle:
            child[idx] = parent[idx]
    return child

def position_based_crossover(parent1, parent2):
    size = len(parent1)
    child = [None]*size

    # Randomly select positions to inherit from parent1
    num_positions = random.randint(1, size)
    positions = random.sample(range(size), num_positions)
    positions.sort()

    # Copy the elements from parent1 at the selected positions to child
    for pos in positions:
        child[pos] = parent1[pos]

    # Fill the remaining positions with the order of elements from parent2
    p2_elements = [elem for elem in parent2 if elem not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = p2_elements.pop(0)
    return child

def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    child = [None]*size

    # Select two crossover points at random
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    # Copy the segment from parent1 to child
    child[cx_point1:cx_point2+1] = parent1[cx_point1:cx_point2+1]

    # Mapping between parent1 and parent2 segments
    mapping = {}
    for i in range(cx_point1, cx_point2+1):
        mapping[parent2[i]] = parent1[i]

    # Fill the remaining positions
    for i in range(size):
        if child[i] is None:
            val = parent2[i]
            while val in mapping:
                val = mapping[val]
            child[i] = val
    return child

def build_edge_table(parent1, parent2):
    edge_table = {}
    all_cities = set(parent1 + parent2)
    for city in all_cities:
        edge_table[city] = set()
    # For parent1
    n = len(parent1)
    for idx, city in enumerate(parent1):
        pred_idx = (idx - 1) % n
        succ_idx = (idx + 1) % n
        pred_city = parent1[pred_idx]
        succ_city = parent1[succ_idx]
        edge_table[city].update([pred_city, succ_city])
    # For parent2
    n = len(parent2)
    for idx, city in enumerate(parent2):
        pred_idx = (idx - 1) % n
        succ_idx = (idx + 1) % n
        pred_city = parent2[pred_idx]
        succ_city = parent2[succ_idx]
        edge_table[city].update([pred_city, succ_city])
    return edge_table

def edge_recombination_crossover(parent1, parent2):
    edge_table = build_edge_table(parent1, parent2)
    child = []
    # Start from the first city of parent1
    current_city = parent1[0]
    unvisited = set(parent1)
    child.append(current_city)
    unvisited.remove(current_city)
    while unvisited:
        # Remove current city from neighbor lists in edge_table
        for neighbors in edge_table.values():
            neighbors.discard(current_city)
        # Get current city's neighbors
        current_neighbors = edge_table[current_city]
        if current_neighbors:
            # Select neighbor with the smallest neighbor list
            min_neighbor_size = min(len(edge_table[neighbor]) for neighbor in current_neighbors)
            candidates = [neighbor for neighbor in current_neighbors if len(edge_table[neighbor]) == min_neighbor_size]
            next_city = random.choice(candidates)
        else:
            # If no neighbors, select randomly from unvisited cities
            next_city = random.choice(list(unvisited))
        child.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    return child

# Helper function to check constraint violations
def constraint_violation(p):
    violations = {'duplicates': 0, 'capacity': 0, 'consecutive_depots': 0}
    
    visited = set()
    total_demand = 0
    consecutive_depot_count = 0
    
    for node in p:
        if node != 0:
            if node in visited:
                violations['duplicates'] += 1
            visited.add(node)
            total_demand += vrp['nodes'][node]['demand']
            if total_demand > vrp['capacity']:
                violations['capacity'] += total_demand - vrp['capacity']
        else:
            if consecutive_depot_count > 0:
                violations['consecutive_depots'] += 1
            consecutive_depot_count += 1
            total_demand = 0
    
    return violations

# Static Penalty Function
def static_penalty(p, C=1000):
    violations = constraint_violation(p)
    penalty = C * (violations['duplicates'] ** 2 + violations['capacity'] ** 2 + violations['consecutive_depots'] ** 2)
    return penalty

# Dynamic Penalty Function
def dynamic_penalty(p, C=1000, alpha=2, generation=1):
    violations = constraint_violation(p)
    penalty = (C * (generation ** alpha)) * (violations['duplicates'] ** 2 + violations['capacity'] ** 2 + violations['consecutive_depots'] ** 2)
    return penalty

# Adaptive Penalty Function
def adaptive_penalty(p, C=1000, lambda_t=1, case1=True):
    violations = constraint_violation(p)
    beta1 = 1.5
    beta2 = 2.0
    
    # Update lambda based on feedback
    if case1:
        lambda_t = lambda_t / beta1
    else:
        lambda_t = lambda_t * beta2

    penalty = lambda_t * C * (violations['duplicates'] ** 2 + violations['capacity'] ** 2 + violations['consecutive_depots'] ** 2)
    return penalty, lambda_t

# Adjust fitness function to include penalty
def fitness_with_penalty(p, C=1000, penalty_type="static", generation=1, lambda_t=1, case1=True):
    # Calculate the basic fitness (distance)
    basic_fitness = distance(vrp['nodes'][0], vrp['nodes'][p[0]])
    for i in range(len(p) - 1):
        prev = vrp['nodes'][p[i]]
        next = vrp['nodes'][p[i + 1]]
        basic_fitness += distance(prev, next)
    basic_fitness += distance(vrp['nodes'][p[-1]], vrp['nodes'][0])

    # Apply penalty based on the penalty type
    if penalty_type == "static":
        penalty = static_penalty(p, C)
    elif penalty_type == "dynamic":
        penalty = dynamic_penalty(p, C, generation=generation)
    elif penalty_type == "adaptive":
        penalty, lambda_t = adaptive_penalty(p, C, lambda_t, case1)
    
    return basic_fitness + penalty, lambda_t

# Main GA loop

popsize = int(sys.argv[1])
iterations = int(sys.argv[2])

pop = []
lambda_t = 1

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
    # Each iteration generates two offspring per pair of parents
    for j in range(int(len(pop) / 2)):
        # Selecting randomly 4 individuals for tournament selection
        parent_ids = set()
        while len(parent_ids) < 4:
            parent_ids.add(random.randint(0, len(pop) - 1))
        parent_ids = list(parent_ids)
        # Selecting 2 parents with the binary tournament
        parent1 = pop[parent_ids[0]] if fitness_with_penalty(pop[parent_ids[0]], penalty_type="adaptive", generation=i, lambda_t=lambda_t)[0] < fitness_with_penalty(pop[parent_ids[1]], penalty_type="adaptive", generation=i, lambda_t=lambda_t)[0] else pop[parent_ids[1]]
        parent2 = pop[parent_ids[2]] if fitness_with_penalty(pop[parent_ids[2]], penalty_type="adaptive", generation=i, lambda_t=lambda_t)[0] < fitness_with_penalty(pop[parent_ids[3]], penalty_type="adaptive", generation=i, lambda_t=lambda_t)[0] else pop[parent_ids[3]]

        # Choose one of the crossover methods by uncommenting the desired function call

        # --------- Edge Recombination (Whitley et al.) ---------
        child1 = edge_recombination_crossover(parent1, parent2)
        child2 = edge_recombination_crossover(parent2, parent1)
        
        # --------- Order Crossover (Davis) ---------
        # child1 = order_crossover_1(parent1, parent2)
        # child2 = order_crossover_1(parent2, parent1)

        # --------- Order Crossover (Syswerda) ---------
        # child1 = order_crossover_2(parent1, parent2)
        # child2 = order_crossover_2(parent2, parent1)

        # --------- Cycle Crossover (Oliver et al.) ---------
        # child1 = cycle_crossover(parent1, parent2)
        # child2 = cycle_crossover(parent2, parent1)

        # --------- Position-Based Crossover (Syswerda) ---------
        # child1 = position_based_crossover(parent1, parent2)
        # child2 = position_based_crossover(parent2, parent1)

        # --------- Partially Mapped Crossover (PMX) (Goldberg and Lingle) ---------
        # child1 = partially_mapped_crossover(parent1, parent2)
        # child2 = partially_mapped_crossover(parent2, parent1)

        next_pop += [child1, child2]

    # # Optional: Doing mutation with a small probability
    # mutation_probability = 0.05  # Adjust as needed
    # for idx in range(len(next_pop)):
    #     if random.random() < mutation_probability:
    #         to_mutate = next_pop[idx]
    #         i1 = random.randint(0, len(to_mutate) - 1)
    #         i2 = random.randint(0, len(to_mutate) - 1)
    #         to_mutate[i1], to_mutate[i2] = to_mutate[i2], to_mutate[i1]
    # Adjusting individuals
    for p in next_pop:
        adjust(p)
    # Updating population generation
    pop = next_pop
    
    current_best_fitness = float('inf')
    for p in pop:
        f, lambda_t = fitness_with_penalty(p, penalty_type="adaptive", generation=i, lambda_t=lambda_t)
        if f < current_best_fitness:
            current_best_fitness = f
    
    print(f"{current_best_fitness:.6f}")
# Selecting the best individual, which is the final solution
better = None
best_fitness = float('inf')
for p in pop:
    f, lambda_t = fitness_with_penalty(p, penalty_type="adaptive", generation=iterations, lambda_t=lambda_t)
    if f < best_fitness:
        best_fitness = f
        better = p

# Print the best solution
print("0")
for node_idx in better:
    x = vrp['nodes'][node_idx]['label']
    if x == "depot":
        print("0")
    else:
        print(x[4:])
print("0")
print(f"{best_fitness:.6f}", end="")