# Cellular Automata Multi-Agent Shape Formation

This document explains the **move_agents_cellular_automata** function in your project, which implements a decentralized, local-rules-based approach for multi-agent shape formation using cellular automata (CA) principles.

---

## Overview

The **cellular automata algorithm** enables agents to self-organize into a target shape using only local information and simple rules. Agents do **not** have global knowledge of the environment or other agents' plans. Instead, they rely on local invitations from settled agents and move step-by-step toward the shape.

---

## Key Concepts

- **Agents:** Entities that move on a grid, each with a unique ID and position.
- **Targets:** Grid cells that define the desired shape to be formed.
- **Obstacles:** Grid cells that agents cannot occupy or move through.
- **Settled Agents:** Agents that have reached a target cell and "lock" their position.
- **Invitations:** Empty target cells adjacent to settled agents, which act as local signals for unsettled agents to move toward.

---

## Algorithm Steps

1. **Initialization**
   - All agents start as "unsettled" (not on a target).
   - The simulation records the initial state.

2. **Main Loop (for each time step)**
   - **a. Invitations Broadcast:**  
     Settled agents (on targets) check their 8 neighbors. If a neighbor is an empty target cell, it is marked as an "invitation."
   - **b. Agent Movement Decision:**  
     - If an agent is already settled, it stays put.
     - If an agent is on a target, it becomes settled and stays put.
     - If there are invitations, each unsettled agent moves to the neighbor cell (or stays) that brings it closest to any invitation (using Manhattan distance).
     - If there are no invitations, agents move toward the closest empty target cell.
   - **c. Conflict Resolution:**  
     If two or more agents want to move to the same cell, all of them stay put (no collisions).
   - **d. Update State:**  
     Agents update their positions. If an agent reaches a target, it becomes settled.
   - **e. Record Step:**  
     The state of all agents is recorded for visualization.

3. **Termination**
   - The simulation stops when all targets are filled by settled agents, or after a maximum number of steps.

---

## Local Information Only

- **Agents only know:**
  - Their own position.
  - Which agents are settled (by observing invitations).
  - The state of their 8 neighboring cells.
- **No randomness** is used; movement is deterministic based on local rules.

---

## Example Rule Summary

- **Settled agents**: Stay put and invite others to adjacent empty targets.
- **Unsettled agents**: Move toward the nearest invitation, or toward the nearest empty target if no invitations exist.
- **Conflicts**: If multiple agents want the same cell, all stay put.

---

## Why This Works

- The algorithm mimics natural swarming and self-assembly, where local signals (invitations) guide agents into the shape.
- No agent needs to know the full target shape or the positions of all other agents.
- The process is robust and scalable for many agents and shapes.

---

## Usage

The function is called as:

```python
move_agents_cellular_automata(grid, agent_positions, target_positions, obstacle_positions)
```

- `grid`: 2D numpy array representing the environment.
- `agent_positions`: List of (row, col) tuples for agent starting positions.
- `target_positions`: List of (row, col) tuples for the shape to form.
- `obstacle_positions`: List of (row, col) tuples for obstacles.

Returns a list of simulation steps, each containing agent positions for visualization.

---

## References

- Cellular Automata: [Wikipedia](https://en.wikipedia.org/wiki/Cellular_automaton)
- Swarm Robotics and Self-Assembly

---

**This approach is ideal for decentralized, scalable, and robust multi-agent shape formation using only local rules.**