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

    return basic_fitness, lambda_t
    # Apply penalty based on the penalty type
    # if penalty_type == "static":
    #     penalty = static_penalty(p, C)
    # elif penalty_type == "dynamic":
    #     penalty = dynamic_penalty(p, C, generation=generation)
    # elif penalty_type == "adaptive":
    #     penalty, lambda_t = adaptive_penalty(p, C, lambda_t, case1)
    
    # return basic_fitness + penalty, lambda_t

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

def bonobo_algorithm(popsize, iterations, pd, pf, pgsm, penalty_type="static"):
    pop = []
    lambda_t = 1

    # Generating random initial population
    for i in range(popsize):
        p = list(range(1, len(vrp['nodes'])))
        random.shuffle(p)
        pop.append(p)
    for p in pop:
        adjust(p)

    # Initialize the alpha bonobo (best solution found so far)
    best_solution = min(pop, key=lambda x: fitness_with_penalty(x, penalty_type=penalty_type)[0])
    best_fitness, lambda_t = fitness_with_penalty(best_solution, penalty_type=penalty_type)

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
                    new_fitness, lambda_t = fitness_with_penalty(new_bonobo, penalty_type=penalty_type, generation=iteration, lambda_t=lambda_t)
                    if new_fitness < fitness_with_penalty(bonobo_p, penalty_type=penalty_type, generation=iteration)[0]:
                        pop[i] = new_bonobo
                        if new_fitness < best_fitness:
                            best_solution = new_bonobo
                            best_fitness = new_fitness

            else:
                # Flag = 0 (perform random exploration)
                new_bonobo = create_new_bonobo_9(bonobo_p)
                # Apply variable boundary limiting conditions
                adjust(new_bonobo)

                # Accept the new solution if it's better
                new_fitness, lambda_t = fitness_with_penalty(new_bonobo, penalty_type=penalty_type, generation=iteration, lambda_t=lambda_t)
                if new_fitness < fitness_with_penalty(bonobo_p, penalty_type=penalty_type, generation=iteration)[0]:
                    pop[i] = new_bonobo
                    if new_fitness < best_fitness:
                        best_solution = new_bonobo
                        best_fitness = new_fitness

        current_best_fitness = float('inf')
        for p in pop:
            f, lambda_t = fitness_with_penalty(p, penalty_type=penalty_type, generation=iteration)
            if f < current_best_fitness:
                current_best_fitness = f
        
        print(f"{current_best_fitness:.6f}")
    return best_solution, best_fitness

# Helper functions to create new bonobos based on the pseudocode
def create_new_bonobo_2(bonobo_p, j):
    new_bonobo = bonobo_p[:]
    idx1, idx2 = random.sample(range(len(new_bonobo)), 2)
    new_bonobo[idx1], new_bonobo[idx2] = new_bonobo[idx2], new_bonobo[idx1]
    return new_bonobo

def create_new_bonobo_3_or_5(bonobo_p, j, best_solution):
    new_bonobo = bonobo_p[:]
    if j < len(best_solution):
        new_bonobo[j] = best_solution[j]
    return new_bonobo

def create_new_bonobo_4_or_6(bonobo_p, j, best_solution):
    new_bonobo = bonobo_p[:]
    if j < len(best_solution):
        new_bonobo[j] = random.choice(best_solution)
    return new_bonobo

def create_new_bonobo_9(bonobo_p):
    new_bonobo = bonobo_p[:]
    idx1, idx2 = random.sample(range(len(new_bonobo)), 2)
    new_bonobo[idx1], new_bonobo[idx2] = new_bonobo[idx2], new_bonobo[idx1]
    return new_bonobo

# Parameters
popsize = int(sys.argv[1])
iterations = int(sys.argv[2])
pd = float(sys.argv[3])  # Example parameter, probability of dispersal
pf = float(sys.argv[4])  # Example parameter, probability of fission-fusion
pgsm = float(sys.argv[5])  # Example parameter, maximum social group size
penalty_type = sys.argv[6]  # static, dynamic, or adaptive

# Run the Bonobo Algorithm
best_solution, best_fitness = bonobo_algorithm(popsize, iterations, pd, pf, pgsm, penalty_type)

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
