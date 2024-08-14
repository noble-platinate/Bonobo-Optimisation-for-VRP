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

def bonobo_algorithm(popsize, iterations, pd, pf, pgsm):
    pop = []

    # Generating random initial population
    for i in range(popsize):
        p = list(range(1, len(vrp['nodes'])))
        random.shuffle(p)
        pop.append(p)
    for p in pop:
        adjust(p)

    # Initialize the alpha bonobo (best solution found so far)
    best_solution = min(pop, key=fitness)
    best_fitness = fitness(best_solution)

    # Running the Bonobo Optimization Algorithm
    for iteration in range(iterations):
        tsgs_max = pgsm * (1 - iteration / iterations)  # Eq. (1)

        for i in range(popsize):
            # Select tsgs based on Eq. (1)
            tsgs = random.uniform(0, tsgs_max)
            # Choose bonobo^p using fission-fusion strategy
            bonobo_p = random.choice(pop)

            # Determine flag based on probability p_f
            if random.random() <= pf:
                # Flag = 1 (perform fission-fusion)
                for j in range(len(bonobo_p)):
                    r = random.random()
                    if r <= pd:
                        # Apply Eq. (2)
                        new_bonobo = create_new_bonobo_2(bonobo_p, j)
                    else:
                        r2 = random.random()
                        if r2 <= pd and j < len(best_solution):
                            new_bonobo = create_new_bonobo_3_or_5(bonobo_p, j, best_solution)
                        else:
                            new_bonobo = create_new_bonobo_4_or_6(bonobo_p, j, best_solution)

                    # Apply variable boundary limiting conditions
                    adjust(new_bonobo)

                    # Accept the new solution if it's better
                    if fitness(new_bonobo) < fitness(bonobo_p):
                        pop[i] = new_bonobo
                        if fitness(new_bonobo) < best_fitness:
                            best_solution = new_bonobo
                            best_fitness = fitness(best_solution)

            else:
                # Flag = 0 (perform random exploration)
                new_bonobo = create_new_bonobo_9(bonobo_p)
                # Apply variable boundary limiting conditions
                adjust(new_bonobo)

                # Accept the new solution if it's better
                if fitness(new_bonobo) < fitness(bonobo_p):
                    pop[i] = new_bonobo
                    if fitness(new_bonobo) < best_fitness:
                        best_solution = new_bonobo
                        best_fitness = fitness(best_solution)

        # Update parameters if needed
        # (In this simplified implementation, we assume no dynamic updates)

    return best_solution, best_fitness

# Helper functions to create new bonobos based on the pseudocode
def create_new_bonobo_2(bonobo_p, j):
    # Implement the logic for Eq. (2) here
    new_bonobo = bonobo_p[:]
    # Randomly swap two nodes as a basic operation
    idx1, idx2 = random.sample(range(len(new_bonobo)), 2)
    new_bonobo[idx1], new_bonobo[idx2] = new_bonobo[idx2], new_bonobo[idx1]
    return new_bonobo

def create_new_bonobo_3_or_5(bonobo_p, j, best_solution):
    # Implement the logic for Eqs. (3) and (5) here
    new_bonobo = bonobo_p[:]
    # Make sure j is within bounds of best_solution
    if j < len(best_solution):
        new_bonobo[j] = best_solution[j]
    return new_bonobo

def create_new_bonobo_4_or_6(bonobo_p, j, best_solution):
    # Implement the logic for Eqs. (4) and (6) here
    new_bonobo = bonobo_p[:]
    # Implement the logic specific to (4) and (6)
    # This is usually a mutation that is slightly different from (3) and (5)
    if j < len(best_solution):
        new_bonobo[j] = random.choice(best_solution)
    return new_bonobo

def create_new_bonobo_9(bonobo_p):
    # Implement the logic for Eq. (9) here
    new_bonobo = bonobo_p[:]
    # Randomly mutate the solution
    idx1, idx2 = random.sample(range(len(new_bonobo)), 2)
    new_bonobo[idx1], new_bonobo[idx2] = new_bonobo[idx2], new_bonobo[idx1]
    return new_bonobo

# Parameters
popsize = int(sys.argv[1])
iterations = int(sys.argv[2])
pd = 0.5  # Example parameter, probability of dispersal
pf = 0.7  # Example parameter, probability of fission-fusion
pgsm = 0.9  # Example parameter, maximum social group size

# Run the Bonobo Algorithm
best_solution, best_fitness = bonobo_algorithm(popsize, iterations, pd, pf, pgsm)

# Printing the solution
print("0")
for node_idx in best_solution:
    x=vrp['nodes'][node_idx]['label']
    if(x=="depot"):
        print("0")
    else: 
        print(x[4:])
print("0")
print(f"{best_fitness:.6f}", end="")