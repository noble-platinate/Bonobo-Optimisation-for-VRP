# Bonobo Optimization for Vehicle Routing Problem (VRP)

This repository implements the Bonobo Optimizer (BO), a novel metaheuristic algorithm inspired by the social behaviors and mating strategies of bonobos, for solving the Vehicle Routing Problem (VRP) with capacity constraints. A comparative analysis with the Genetic Algorithm (GA) is also provided to evaluate the performance of BO in optimizing delivery routes.

## Overview

**Vehicle Routing Problem (VRP)**:  
VRP involves determining optimal delivery routes for vehicles to serve a set of locations under constraints such as capacity and time. This problem is NP-hard, and exact solutions are computationally infeasible for large-scale instances.

**Bonobo Optimizer (BO)**:  
BO simulates bonobo social dynamics, including fission-fusion grouping and varied mating strategies. This algorithm balances exploration (diversifying solutions) and exploitation (refining the best solutions) dynamically, ensuring effective navigation of the solution space.

**Genetic Algorithm (GA)**:  
GA emulates the process of natural selection through selection, crossover, and mutation. While efficient, GA faces challenges in convergence speed and parameter tuning for VRP.

## Key Features

- Implementation of Bonobo Optimizer (BO) for VRP.
- Comparative analysis with Genetic Algorithm (GA).
- Simulation of a VRP scenario with capacity constraints in a simulated city environment.
- Penalty mechanisms to handle infeasible solutions.

## Repository Contents

- **`bonobo_optimizer.py`**: Core implementation of the Bonobo Optimizer.
- **`genetic_algorithm.py`**: Implementation of the Genetic Algorithm for comparison.
- **`vrp_simulation.py`**: Code for setting up the VRP instance and running simulations.
- **`data/`**: Input data for the VRP simulation (e.g., coordinates, demands).
- **`results/`**: Output results from simulations, including convergence plots and cost analysis.
- **`MTP_Initial_Draft.pdf`**: A detailed project report explaining the methodology, experiments, and results.

## Methodology

1. **Formulation**:  
   The VRP is modeled mathematically with:
   - Decision variables for route selection.
   - An objective function minimizing total travel distance.
   - Constraints for vehicle capacity, demand fulfillment, and depot returns.

2. **Bonobo Optimizer**:  
   - **Initialization**: Random initialization of a population of solutions.
   - **Positive Phase**: Exploitation near the best solution (alpha bonobo).
   - **Negative Phase**: Exploration of new regions in the solution space.
   - **Mating Strategies**: Includes promiscuous, restrictive, consortship, and extra-group mating.
   - **Adaptive Parameters**: Dynamic adjustment of algorithm parameters.

3. **Genetic Algorithm**:  
   - **Selection**: Tournament selection of parent chromosomes.
   - **Crossover**: Edge Recombination Crossover (ERC) for route connectivity.
   - **Mutation**: Operations like swap, inversion, and insertion for diversity.

4. **Evaluation**:  
   - Cost performance: Minimizing the total travel distance.
   - Handling infeasible solutions using static, dynamic, and adaptive penalty functions.

## Results

- **Performance**: BO consistently outperforms GA in minimizing total travel distance for VRP.
- **Convergence**: BO demonstrates faster convergence to optimal solutions after sufficient iterations.
- **Trade-offs**: BO requires higher computational time due to its complex dynamics.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/noble-platinate/Bonobo-Optimisation-for-VRP.git
   cd Bonobo-Optimisation-for-VRP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute simulations:
   ```bash
   python vrp_simulation.py
   ```

4. Review results in the `results/` directory.

## Reference

For detailed explanations, methodologies, and results, please refer to the [MTP Initial Draft PDF](MTP_Initial_Draft.pdf) included in the repository.
