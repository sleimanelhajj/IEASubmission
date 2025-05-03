# Genetic Algorithm for Agent-Target Assignment

This document describes the Genetic Algorithm (GA) implemented in [`run_genetic_algorithm`](Project2/backend/app.py) within the [Project2/backend/app.py](Project2/backend/app.py) file. The GA aims to find an optimal assignment of agents to targets that minimizes the total movement cost, measured primarily by the number of simulation steps required for all agents to reach their assigned targets.

## Core Concepts

1.  **Chromosome Representation**:
    *   Each chromosome represents a potential solution, specifically an assignment of targets to agents.
    *   It's implemented as a list where the index corresponds to the agent ID, and the value at that index corresponds to the target ID assigned to that agent.
    *   The list contains a permutation of target indices. If there are more agents than targets (`num_agents > num_targets`), some agents might be assigned `None`, indicating they have no target. If there are more targets than agents, only a subset of targets will be assigned. See [`random_chromosome`](Project2/backend/app.py).

2.  **Fitness Evaluation**:
    *   The [`evaluate_fitness`](Project2/backend/app.py) function determines the quality (fitness) of a given chromosome (assignment).
    *   It simulates the movement of agents based on the assignments defined by the chromosome. This simulation uses a simplified, non-animated movement logic with collision detection and resolution (similar to the final animation loop).
    *   The fitness score is calculated based on the number of steps taken until all agents reach their targets or a maximum step limit (`max_steps`) is hit.
    *   A significant penalty (`max_steps`) is added for each agent that fails to reach its assigned target within the simulation limit.
    *   **Lower fitness scores are better**, indicating fewer steps and fewer penalties.

3.  **Genetic Operators**:
    *   **Selection**: Tournament selection is used. A small group of individuals is randomly chosen from the population, and the one with the best (lowest) fitness score is selected as a parent.
    *   **Crossover**: Order Crossover (OX) is implemented in [`crossover`](Project2/backend/app.py). This method is suitable for permutation-based chromosomes. It preserves the relative order of elements from the parents while creating offspring.
    *   **Mutation**: Swap mutation is used in [`mutate`](Project2/backend/app.py). Two random positions in the chromosome are selected, and their values (target assignments) are swapped.
    *   **Elitism**: A small number of the best-performing individuals from the current generation are directly carried over to the next generation to ensure the best solutions found so far are not lost.

## Algorithm Flow

1.  **Initialization**: A population of `population_size` random chromosomes is generated using [`random_chromosome`](Project2/backend/app.py).
2.  **Evolution Loop**: The algorithm iterates for a fixed number of `generations`. In each generation:
    *   The fitness of every chromosome in the current population is calculated using [`evaluate_fitness`](Project2/backend/app.py).
    *   The population is sorted based on fitness scores.
    *   A new generation is created:
        *   The top individuals (elitism) are copied directly to the new generation.
        *   The remaining spots are filled by creating offspring:
            *   Parents are selected using tournament selection.
            *   Offspring are generated using the [`crossover`](Project2/backend/app.py) operator.
            *   The [`mutate`](Project2/backend/app.py) operator is applied to the offspring with a probability defined by `mutation_rate`.
    *   The new generation replaces the old population.
3.  **Termination**: After the final generation, the chromosome with the best fitness score in the population is selected as the optimal solution found by the GA.
4.  **Result**: The best chromosome dictates the final agent-to-target assignments. The function then proceeds to simulate and animate the movement of agents based on this optimal assignment, using the full collision-aware movement logic.

This GA provides a heuristic approach to find a near-optimal assignment, balancing the complexity of multi-agent pathfinding with the need for an efficient solution.