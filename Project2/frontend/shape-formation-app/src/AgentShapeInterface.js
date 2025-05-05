import React, { useState, useEffect } from "react";
import "./AgentShapeInterface.css";

const AgentShapeInterface = () => {
  // Main state variables
  const [mode, setMode] = useState("selection");
  const [selectedPredefinedShape, setSelectedPredefinedShape] = useState(null);
  const [placementMode, setPlacementMode] = useState("none");
  const [gridSize, setGridSize] = useState(15);
  const [grid, setGrid] = useState([]);
  const [message, setMessage] = useState("Select a mode to begin");
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [lastCellInteracted, setLastCellInteracted] = useState(null);
  const [dragMode, setDragMode] = useState(null);

  // Backend configuration options
  const [algorithm, setAlgorithm] = useState("inside-out");
  const [agentTopology, setAgentTopology] = useState("8-directional");
  const [agentCount, setAgentCount] = useState(5);
  const [agentSpeed, setAgentSpeed] = useState("medium");

  // Simulation state
  const [isSimulating, setIsSimulating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showQValues, setShowQValues] = useState(false);
  const [showAgentIds, setShowAgentIds] = useState(true);
  const [showPathFinding, setShowPathFinding] = useState(true);
  const [highlightTargets, setHighlightTargets] = useState(true);

  // Q-learning episode/step state
  const [allEpisodes, setAllEpisodes] = useState([]);
  const [exploitationSteps, setExploitationSteps] = useState([]);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [currentEpisodeStep, setCurrentEpisodeStep] = useState(0);
  const [playbackTimeout, setPlaybackTimeout] = useState(null);

  // Classic (non-Q) simulation state
  const [simulationSteps, setSimulationSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playbackIntervalId, setPlaybackIntervalId] = useState(null);

  // Shared
  const [isPlaying, setIsPlaying] = useState(false);
  const [qValues, setQValues] = useState([]);
  const [agentPositions, setAgentPositions] = useState([]);

  // Grid initialization
  useEffect(() => {
    const newGrid = Array(gridSize)
      .fill()
      .map(() =>
        Array(gridSize)
          .fill()
          .map(() => ({ type: "empty" }))
      );
    setGrid(newGrid);
  }, [gridSize]);

  // Update agent positions for current step/episode
  useEffect(() => {
    if (algorithm === "qlearning") {
      if (
        allEpisodes.length > 0 &&
        currentEpisode < allEpisodes.length &&
        currentEpisodeStep < (allEpisodes[currentEpisode]?.length || 0)
      ) {
        setAgentPositions(
          allEpisodes[currentEpisode][currentEpisodeStep].agents
        );
      } else if (
        allEpisodes.length > 0 &&
        currentEpisode === allEpisodes.length &&
        currentEpisodeStep < exploitationSteps.length
      ) {
        setAgentPositions(exploitationSteps[currentEpisodeStep].agents);
      } else {
        setAgentPositions([]);
      }
    } else {
      if (simulationSteps.length > 0 && currentStep < simulationSteps.length) {
        setAgentPositions(simulationSteps[currentStep].agents);
      } else {
        setAgentPositions([]);
      }
    }
  }, [
    algorithm,
    allEpisodes,
    exploitationSteps,
    currentEpisode,
    currentEpisodeStep,
    simulationSteps,
    currentStep,
  ]);

  // Mouse up cleanup
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsMouseDown(false);
      setLastCellInteracted(null);
      setDragMode(null);
    };
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, []);

  // Predefined shapes
  const predefinedShapes = {
    square: [
      [0, 0],
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 0],
      [1, 3],
      [2, 0],
      [2, 3],
      [3, 0],
      [3, 1],
      [3, 2],
      [3, 3],
    ],
    triangle: [
      [0, 2],
      [1, 1],
      [1, 3],
      [2, 0],
      [2, 1],
      [2, 2],
      [2, 3],
      [2, 4],
    ],
    circle: [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 0],
      [1, 4],
      [2, 0],
      [2, 4],
      [3, 0],
      [3, 4],
      [4, 1],
      [4, 2],
      [4, 3],
    ],
  };

  // Apply predefined shape to grid
  const applyPredefinedShape = (shape) => {
    setSelectedPredefinedShape(shape);
    const newGrid = [
      ...grid.map((row) => [...row.map((cell) => ({ ...cell }))]),
    ];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        if (newGrid[i][j].type === "shape") {
          newGrid[i][j].type = "empty";
        }
      }
    }
    const centerOffset = Math.floor(gridSize / 2) - 2;
    let shapeCellCount = 0;
    predefinedShapes[shape].forEach(([x, y]) => {
      const newX = x + centerOffset;
      const newY = y + centerOffset;
      if (newX >= 0 && newX < gridSize && newY >= 0 && newY < gridSize) {
        newGrid[newX][newY].type = "shape";
        shapeCellCount++;
      }
    });
    setGrid(newGrid);
    setAgentCount(shapeCellCount);
    setMessage(
      `${
        shape.charAt(0).toUpperCase() + shape.slice(1)
      } shape placed with ${shapeCellCount} agents. You can now add obstacles.`
    );
  };

  // Reset grid
  const resetGrid = () => {
    const newGrid = Array(gridSize)
      .fill()
      .map(() =>
        Array(gridSize)
          .fill()
          .map(() => ({ type: "empty" }))
      );
    setGrid(newGrid);
    setSelectedPredefinedShape(null);
    setPlacementMode("none");
    setMessage("Grid reset");
    setAgentCount(1);
    setIsSimulating(false);
    setAllEpisodes([]);
    setExploitationSteps([]);
    setCurrentEpisode(0);
    setCurrentEpisodeStep(0);
    setSimulationSteps([]);
    setCurrentStep(0);
    setAgentPositions([]);
    setQValues([]);
  };

  // Change mode
  const changeMode = (newMode) => {
    setMode(newMode);
    resetGrid();
    if (newMode === "selection") {
      setMessage("Select a mode to begin");
    } else if (newMode === "predefined") {
      setMessage("Select a predefined shape");
    } else if (newMode === "custom") {
      setMessage("Create your custom shape by placing green blocks");
      setPlacementMode("shape");
    }
  };

  // Run simulation (connecting to Python backend)
  const runSimulation = async () => {
    const hasShape = grid.some((row) =>
      row.some((cell) => cell.type === "shape")
    );
    if (!hasShape) {
      setMessage("Error: No shape defined. Please create a shape first.");
      return;
    }
    setMessage("Sending data to backend for processing...");
    setIsSimulating(true);
    setIsLoading(true);

    const gridData = grid.map((row) =>
      row.map((cell) => {
        if (cell.type === "empty") return 0;
        if (cell.type === "shape") return 1;
        if (cell.type === "obstacle") return 2;
        return 0;
      })
    );
    const configData = {
      algorithm,
      agentTopology,
      agentCount,
      agentSpeed,
      gridSize,
      showPathFinding,
    };

    try {
      const response = await fetch("http://localhost:8000/run_simulation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gridData, configData }),
      });
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      const result = await response.json();

      if (algorithm === "qlearning") {
        setAllEpisodes(result.episodes || []);
        setExploitationSteps(result.exploitation || []);
        setQValues(result.q_values || []);
        setCurrentEpisode(0);
        setCurrentEpisodeStep(0);
        setSimulationSteps([]);
        setCurrentStep(0);
        setMessage("Simulation received! Showing learning episodes...");
      } else {
        setSimulationSteps(result.steps || []);
        setCurrentStep(0);
        setAllEpisodes([]);
        setExploitationSteps([]);
        setCurrentEpisode(0);
        setCurrentEpisodeStep(0);
        setQValues([]);
        setMessage("Simulation received! Showing steps...");
      }
    } catch (error) {
      setMessage(`Error: ${error.message}. Is the Python backend running?`);
      setIsSimulating(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Play simulation (Q-learning or classic)
  const playSimulation = () => {
    if (algorithm === "qlearning") {
      if (allEpisodes.length === 0) return;
      if (isPlaying) return;
      setIsPlaying(true);

      let ep = currentEpisode;
      let st = currentEpisodeStep;
      const speed =
        agentSpeed === "slow" ? 1000 : agentSpeed === "medium" ? 500 : 50;

      const playNext = () => {
        if (ep < allEpisodes.length) {
          if (st < allEpisodes[ep].length - 1) {
            setCurrentEpisode(ep);
            setCurrentEpisodeStep(++st);
            setPlaybackTimeout(setTimeout(playNext, speed));
          } else {
            ep++;
            st = 0;
            setCurrentEpisode(ep);
            setCurrentEpisodeStep(st);
            setPlaybackTimeout(setTimeout(playNext, speed));
          }
        } else if (
          ep === allEpisodes.length &&
          st < exploitationSteps.length - 1
        ) {
          setCurrentEpisode(ep);
          setCurrentEpisodeStep(++st);
          setPlaybackTimeout(setTimeout(playNext, speed));
        } else {
          setIsPlaying(false);
          setMessage("Simulation complete!");
        }
      };

      playNext();
    } else {
      if (simulationSteps.length === 0) return;
      if (isPlaying) return;
      setIsPlaying(true);
      setCurrentStep((prev) =>
        prev === simulationSteps.length - 1 ? 0 : prev
      );
      const speed =
        agentSpeed === "slow" ? 1000 : agentSpeed === "medium" ? 500 : 50;

      const intervalId = setInterval(() => {
        setCurrentStep((prevStep) => {
          if (prevStep < simulationSteps.length - 1) {
            return prevStep + 1;
          } else {
            clearInterval(intervalId);
            setIsPlaying(false);
            setPlaybackIntervalId(null);
            setMessage("Simulation complete!");
            return prevStep;
          }
        });
      }, speed);

      setPlaybackIntervalId(intervalId);
    }
  };

  // Pause simulation
  const pauseSimulation = () => {
    if (algorithm === "qlearning") {
      if (playbackTimeout) {
        clearTimeout(playbackTimeout);
        setPlaybackTimeout(null);
        setIsPlaying(false);
        setMessage("Simulation paused");
      }
    } else {
      if (playbackIntervalId) {
        clearInterval(playbackIntervalId);
        setPlaybackIntervalId(null);
        setIsPlaying(false);
        setMessage("Simulation paused");
      }
    }
  };

  // Stop simulation and return to editing
  const stopSimulation = () => {
    if (algorithm === "qlearning") {
      if (playbackTimeout) {
        clearTimeout(playbackTimeout);
        setPlaybackTimeout(null);
      }
      setIsPlaying(false);
      setIsSimulating(false);
      setAllEpisodes([]);
      setExploitationSteps([]);
      setCurrentEpisode(0);
      setCurrentEpisodeStep(0);
      setAgentPositions([]);
      setQValues([]);
      setSimulationSteps([]);
      setCurrentStep(0);
    } else {
      if (playbackIntervalId) {
        clearInterval(playbackIntervalId);
        setPlaybackIntervalId(null);
      }
      setIsPlaying(false);
      setIsSimulating(false);
      setSimulationSteps([]);
      setCurrentStep(0);
      setAgentPositions([]);
      setQValues([]);
      setAllEpisodes([]);
      setExploitationSteps([]);
      setCurrentEpisode(0);
      setCurrentEpisodeStep(0);
    }
    setMessage("Simulation stopped. You can modify the grid now.");
  };

  // Algorithm description
  const getAlgorithmDescription = () => {
    switch (algorithm) {
      case "inside-out":
        return "This algorithm uses an inside-out approach where inner shape cells are filled first, then outer cells. It prioritizes filling centrally located targets first.";
      case "leader-follower":
        return "In this approach, one agent (the leader) moves to its target while others follow behind. After the leader reaches its target, other agents move independently.";
      case "centralized":
        return "The leader agent first visits all target positions one by one. As it visits each target, the closest available agent is assigned to stay at that position.";
      case "Genetic-Algorithm":
        return "This algorithm uses a genetic approach to optimize the shape formation. It evolves a population of solutions over generations, selecting the best ones based on fitness criteria.";
      default:
        return "";
    }
  };

  // Mouse click and drag event handling
  const handleMouseDown = (row, col) => {
    if (isSimulating) return;
    setIsMouseDown(true);
    setLastCellInteracted({ row, col });
    const cellType = grid[row][col].type;
    if (mode === "custom" && placementMode === "shape") {
      if (cellType === "shape") setDragMode("unselect");
      else if (cellType === "empty") setDragMode("select");
    } else if (placementMode === "obstacle") {
      if (cellType === "obstacle") setDragMode("unselect");
      else if (cellType === "empty") setDragMode("select");
    }
    handleCellInteraction(row, col);
  };

  const handleMouseUp = () => {
    setIsMouseDown(false);
    setLastCellInteracted(null);
    setDragMode(null);
  };

  const handleMouseEnter = (row, col) => {
    if (isSimulating) return;
    if (
      isMouseDown &&
      dragMode !== null &&
      (lastCellInteracted === null ||
        lastCellInteracted.row !== row ||
        lastCellInteracted.col !== col)
    ) {
      setLastCellInteracted({ row, col });
      handleCellInteraction(row, col);
    }
  };

  // Common function to handle cell interactions (both clicks and drags)
  const handleCellInteraction = (row, col) => {
    const newGrid = [...grid.map((r) => [...r.map((cell) => ({ ...cell }))])];
    if (mode === "predefined") {
      if (placementMode === "obstacle") {
        if (newGrid[row][col].type === "shape") return;
        if (dragMode === "select") {
          if (newGrid[row][col].type !== "obstacle") {
            newGrid[row][col].type = "obstacle";
            setMessage("Obstacle placed");
          }
        } else if (dragMode === "unselect") {
          if (newGrid[row][col].type === "obstacle") {
            newGrid[row][col].type = "empty";
            setMessage("Obstacle removed");
          }
        }
      }
    } else if (mode === "custom") {
      if (placementMode === "shape") {
        if (newGrid[row][col].type === "obstacle") return;
        if (dragMode === "select") {
          if (newGrid[row][col].type !== "shape") {
            newGrid[row][col].type = "shape";
            setMessage("Shape block placed");
            setAgentCount((prevCount) => prevCount + 1);
          }
        } else if (dragMode === "unselect") {
          if (newGrid[row][col].type === "shape") {
            newGrid[row][col].type = "empty";
            setMessage("Shape block removed");
            setAgentCount((prevCount) => Math.max(0, prevCount - 1));
          }
        }
      } else if (placementMode === "obstacle") {
        if (newGrid[row][col].type === "shape") return;
        if (dragMode === "select") {
          if (newGrid[row][col].type !== "obstacle") {
            newGrid[row][col].type = "obstacle";
            setMessage("Obstacle placed");
          }
        } else if (dragMode === "unselect") {
          if (newGrid[row][col].type === "obstacle") {
            newGrid[row][col].type = "empty";
            setMessage("Obstacle removed");
          }
        }
      }
    }
    setGrid(newGrid);
  };

  // Episode controls UI (Q-learning)
  const episodeControls = algorithm === "qlearning" &&
    isSimulating &&
    allEpisodes.length > 0 && (
      <div className="episode-controls" style={{ margin: "10px 0" }}>
        <label>Episode:</label>
        <input
          type="number"
          min={0}
          max={allEpisodes.length}
          value={currentEpisode}
          onChange={(e) => {
            let val = Number(e.target.value);
            if (val > allEpisodes.length) val = allEpisodes.length;
            setCurrentEpisode(val);
            setCurrentEpisodeStep(0);
          }}
          disabled={isPlaying}
          style={{ width: 60, marginLeft: 5, marginRight: 10 }}
        />
        <span>
          {currentEpisode < allEpisodes.length
            ? `Step ${currentEpisodeStep + 1} of ${
                allEpisodes[currentEpisode]?.length || 0
              }`
            : `Exploitation Step ${currentEpisodeStep + 1} of ${
                exploitationSteps.length
              }`}
        </span>
        <button
          onClick={() =>
            setCurrentEpisodeStep(Math.max(0, currentEpisodeStep - 1))
          }
          disabled={currentEpisodeStep === 0 || isPlaying}
          style={{ marginLeft: 10 }}
        >
          Prev Step
        </button>
        <button
          onClick={() => {
            if (currentEpisode < allEpisodes.length) {
              setCurrentEpisodeStep(
                Math.min(
                  allEpisodes[currentEpisode].length - 1,
                  currentEpisodeStep + 1
                )
              );
            } else {
              setCurrentEpisodeStep(
                Math.min(exploitationSteps.length - 1, currentEpisodeStep + 1)
              );
            }
          }}
          disabled={
            (currentEpisode < allEpisodes.length &&
              currentEpisodeStep ===
                (allEpisodes[currentEpisode]?.length || 1) - 1) ||
            (currentEpisode === allEpisodes.length &&
              currentEpisodeStep === exploitationSteps.length - 1) ||
            isPlaying
          }
        >
          Next Step
        </button>
        <button
          onClick={() => {
            if (currentEpisode < allEpisodes.length) {
              setCurrentEpisode(
                Math.min(allEpisodes.length, currentEpisode + 1)
              );
              setCurrentEpisodeStep(0);
            }
          }}
          disabled={currentEpisode === allEpisodes.length || isPlaying}
          style={{ marginLeft: 10 }}
        >
          Next Episode
        </button>
      </div>
    );

  // Step controls UI (classic)
  const stepControls = algorithm !== "qlearning" &&
    isSimulating &&
    simulationSteps.length > 0 && (
      <div className="flex items-center justify-between mt-2">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0 || isPlaying}
          className={`btn btn-blue ${
            currentStep === 0 || isPlaying ? "btn-disabled" : ""
          }`}
        >
          Previous
        </button>
        <span className="mx-2">
          Step {currentStep + 1} of {simulationSteps.length}
        </span>
        <button
          onClick={() =>
            setCurrentStep(
              Math.min(simulationSteps.length - 1, currentStep + 1)
            )
          }
          disabled={currentStep === simulationSteps.length - 1 || isPlaying}
          className={`btn btn-blue ${
            currentStep === simulationSteps.length - 1 || isPlaying
              ? "btn-disabled"
              : ""
          }`}
        >
          Next
        </button>
      </div>
    );

  // --- Render ---
  return (
    <div className="app-container">
      <div className="header">
        <h1>Multi-Agent Shape Formation</h1>
      </div>
      <div className="message-bar">
        {message}
        {isLoading && <span className="loading"> Loading...</span>}
      </div>
      <div className="main-content">
        <div className="sidebar">
          <h2>Mode Selection</h2>
          <button
            onClick={() => changeMode("selection")}
            disabled={isSimulating}
            className={`btn ${mode === "selection" ? "btn-blue" : "btn-gray"} ${
              isSimulating ? "btn-disabled" : ""
            }`}
          >
            Back to Selection
          </button>
          <button
            onClick={() => changeMode("predefined")}
            disabled={isSimulating}
            className={`btn ${
              mode === "predefined" ? "btn-blue" : "btn-gray"
            } ${isSimulating ? "btn-disabled" : ""}`}
          >
            Predefined Shapes
          </button>
          <button
            onClick={() => changeMode("custom")}
            disabled={isSimulating}
            className={`btn ${mode === "custom" ? "btn-blue" : "btn-gray"} ${
              isSimulating ? "btn-disabled" : ""
            }`}
          >
            Custom Shapes
          </button>
          {mode === "predefined" && !isSimulating && (
            <>
              <h3 className="mt-4">Select Shape</h3>
              <div className="mb-4">
                <button
                  onClick={() => applyPredefinedShape("square")}
                  className={`btn btn-full ${
                    selectedPredefinedShape === "square"
                      ? "btn-green"
                      : "btn-gray"
                  }`}
                >
                  Square
                </button>
                <button
                  onClick={() => applyPredefinedShape("triangle")}
                  className={`btn btn-full ${
                    selectedPredefinedShape === "triangle"
                      ? "btn-green"
                      : "btn-gray"
                  }`}
                >
                  Triangle
                </button>
                <button
                  onClick={() => applyPredefinedShape("circle")}
                  className={`btn btn-full ${
                    selectedPredefinedShape === "circle"
                      ? "btn-green"
                      : "btn-gray"
                  }`}
                >
                  Circle
                </button>
              </div>
            </>
          )}
          {(mode === "predefined" || mode === "custom") && !isSimulating && (
            <>
              <h3>Grid Controls</h3>
              <div className="mb-4">
                {mode === "custom" && (
                  <button
                    onClick={() => setPlacementMode("shape")}
                    className={`btn btn-full ${
                      placementMode === "shape" ? "btn-green" : "btn-gray"
                    }`}
                  >
                    Place Shape Blocks
                  </button>
                )}
                <button
                  onClick={() => setPlacementMode("obstacle")}
                  className={`btn btn-full ${
                    placementMode === "obstacle" ? "btn-red" : "btn-gray"
                  }`}
                >
                  Place Obstacles
                </button>
                <button onClick={resetGrid} className="btn btn-full btn-yellow">
                  Reset Grid
                </button>
                <div className="form-group">
                  <label className="form-label">Grid Size</label>
                  <select
                    value={gridSize}
                    onChange={(e) => setGridSize(parseInt(e.target.value, 10))}
                    className="form-select"
                  >
                    <option value={10}>10 x 10</option>
                    <option value={15}>15 x 15</option>
                    <option value={20}>20 x 20</option>
                    <option value={25}>25 x 25</option>
                  </select>
                </div>
              </div>
              <button
                onClick={runSimulation}
                disabled={isSimulating || isLoading}
                className={`btn btn-full btn-blue mt-auto ${
                  isSimulating || isLoading ? "btn-disabled" : ""
                }`}
              >
                Run Simulation
              </button>
            </>
          )}
          {isSimulating && (
            <>
              <h3 className="mt-4">Simulation Controls</h3>
              <div className="mb-4">
                {!isPlaying ? (
                  <button
                    onClick={playSimulation}
                    className="btn btn-full btn-green"
                  >
                    Play Simulation
                  </button>
                ) : (
                  <button
                    onClick={pauseSimulation}
                    className="btn btn-full btn-yellow"
                  >
                    Pause Simulation
                  </button>
                )}
                {algorithm === "qlearning" ? episodeControls : stepControls}
                <button
                  onClick={stopSimulation}
                  className="btn btn-full btn-red mt-4"
                >
                  Stop & Edit
                </button>
              </div>
            </>
          )}
        </div>
        <div className="grid-container">
          {mode === "selection" ? (
            <div className="welcome-screen">
              <h2>Welcome to Multi-Agent Shape Formation</h2>
              <p>Select a mode to get started:</p>
              <div className="welcome-buttons">
                <button
                  onClick={() => changeMode("predefined")}
                  className="welcome-btn welcome-btn-blue"
                >
                  Predefined Shapes
                </button>
                <button
                  onClick={() => changeMode("custom")}
                  className="welcome-btn welcome-btn-green"
                >
                  Custom Shapes
                </button>
              </div>
            </div>
          ) : (
            <div
              className="grid"
              style={{
                gridTemplateColumns: `repeat(${gridSize}, 24px)`,
                gridTemplateRows: `repeat(${gridSize}, 24px)`,
              }}
            >
              {grid.map((row, rowIndex) =>
                row.map((cell, colIndex) => {
                  const isPlacementDisallowed =
                    (placementMode === "obstacle" && cell.type === "shape") ||
                    (placementMode === "shape" && cell.type === "obstacle");
                  const agent = agentPositions?.find(
                    (a) => a.x === rowIndex && a.y === colIndex
                  );
                  const isTargetCell =
                    highlightTargets && cell.type === "shape" && isSimulating;
                  let cellClassName = "cell ";
                  if (agent) {
                    if (algorithm === "qlearning") {
                      cellClassName += " cell-agent-black";
                    } else {
                      cellClassName += " cell-agent";
                    }
                  } else if (cell.type === "shape") {
                    cellClassName += isTargetCell
                      ? "cell-target-highlight"
                      : "cell-shape";
                  } else if (cell.type === "obstacle") {
                    cellClassName += "cell-obstacle";
                  } else {
                    cellClassName += "cell-empty";
                  }
                  if (
                    !isSimulating &&
                    mode !== "selection" &&
                    placementMode !== "none" &&
                    isPlacementDisallowed
                  ) {
                    cellClassName += " not-allowed";
                  }
                  // Q-value heatmap
                  let qValue = null;
                  if (
                    showQValues &&
                    qValues.length > 0 &&
                    qValues[rowIndex] &&
                    typeof qValues[rowIndex][colIndex] === "number"
                  ) {
                    qValue = qValues[rowIndex][colIndex];
                    const norm = Math.max(0, Math.min(1, (qValue + 0.01) / 1));
                    cellClassName += " cell-qvalue";
                    var qStyle = {
                      background: `rgba(255, 0, 0, ${norm})`,
                    };
                  } else {
                    var qStyle = {};
                  }
                  return (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                      onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                      className={cellClassName}
                      style={qStyle}
                    >
                      {agent && showAgentIds && (
                        <span className="agent-id">{agent.id}</span>
                      )}
                      {!isSimulating &&
                        mode !== "selection" &&
                        placementMode !== "none" &&
                        isPlacementDisallowed && (
                          <div className="not-allowed-indicator">
                            <div className="not-allowed-symbol">
                              <span>×</span>
                            </div>
                          </div>
                        )}
                      {
                        showPathFinding &&
                          isSimulating &&
                          false /* pathfinding not shown in Q-learning */
                      }
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
        {(mode === "predefined" || mode === "custom") && (
          <div className="sidebar">
            <h2>Simulation Settings</h2>
            <div>
              <div className="form-group">
                <label className="form-label">Algorithm</label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  disabled={isSimulating}
                  className={`form-select ${
                    isSimulating ? "btn-disabled" : ""
                  }`}
                >
                  <option value="inside-out">Inside-Out</option>
                  <option value="leader-follower">Leader-Follower</option>
                  <option value="centralized">Centralized</option>
                  <option value="Genetic-Algorithm">Genetic Algorithm</option>
                  <option value="cellular-automata">Cellular Automata</option>
                  <option value="minimax">Minimax</option>
                  <option value="expectimax">Expectimax</option>
                  <option value="minimax-adv">Minimax-adc</option>
                  <option value="gradient-field">
                    Gradient Field (Distributed)
                  </option>
                  <option value="qlearning">Q learning</option>
                  <option value="deep">deep learning</option>
                </select>
                <p className="form-text">{getAlgorithmDescription()}</p>
              </div>
              <div className="form-group">
                <label className="form-label">Agent Movement</label>
                <select
                  value={agentTopology}
                  onChange={(e) => setAgentTopology(e.target.value)}
                  disabled={true}
                  className="form-select btn-disabled"
                >
                  <option value="8-directional">
                    8-Directional (NSEW + Diagonals)
                  </option>
                </select>
                <p className="form-text">
                  Your algorithms use 8-directional movement
                </p>
              </div>
              <div className="form-group">
                <label className="form-label">Number of Agents</label>
                <input
                  type="range"
                  min="1"
                  max={gridSize * 3}
                  value={agentCount}
                  onChange={(e) => setAgentCount(parseInt(e.target.value, 10))}
                  disabled={isSimulating}
                  className={isSimulating ? "btn-disabled" : ""}
                />
                <div className="flex justify-between">
                  <span>1</span>
                  <span>{agentCount}</span>
                  <span>{gridSize * 3}</span>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Animation Speed</label>
                <select
                  value={agentSpeed}
                  onChange={(e) => setAgentSpeed(e.target.value)}
                  className="form-select"
                >
                  <option value="slow">Slow</option>
                  <option value="medium">Medium</option>
                  <option value="fast">Fast</option>
                </select>
              </div>
              <div className="form-group">
                <h3>Visualization Options</h3>
                <div>
                  <div className="checkbox-container">
                    <input
                      type="checkbox"
                      id="showQValues"
                      checked={showQValues}
                      onChange={(e) => setShowQValues(e.target.checked)}
                    />
                    <span>Show Q-Value Heatmap</span>
                  </div>
                  <div className="checkbox-container">
                    <input
                      type="checkbox"
                      id="showPathFinding"
                      checked={showPathFinding}
                      onChange={(e) => setShowPathFinding(e.target.checked)}
                    />
                    <span>Show Path Finding</span>
                  </div>
                  <div className="checkbox-container">
                    <input
                      type="checkbox"
                      id="showAgentIds"
                      checked={showAgentIds}
                      onChange={(e) => setShowAgentIds(e.target.checked)}
                    />
                    <span>Show Agent IDs</span>
                  </div>
                  <div className="checkbox-container">
                    <input
                      type="checkbox"
                      id="highlightTargets"
                      checked={highlightTargets}
                      onChange={(e) => setHighlightTargets(e.target.checked)}
                    />
                    <span>Highlight Target Cells</span>
                  </div>
                </div>
              </div>
              <div className="info-box mt-4">
                <h3 className="mb-2">Algorithm Details</h3>
                <p className="info-text">{getAlgorithmDescription()}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentShapeInterface;
