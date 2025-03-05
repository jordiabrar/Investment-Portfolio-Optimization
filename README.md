# Portfolio Optimization with Machine Learning in Julia

## Overview
This project implements portfolio optimization using three different approaches:
1. **Mean-Variance Optimization** using JuMP.jl to find the optimal asset allocation.
2. **Reinforcement Learning (Q-Learning)** for asset selection in a simple 2-asset scenario.
3. **Machine Learning (Regression)** using MLJ.jl to predict asset returns based on historical features.

## Technologies Used
- **JuMP.jl**: Mathematical optimization
- **OSQP.jl**: Quadratic programming solver
- **MLJ.jl**: Machine learning framework
- **DataFrames.jl**: Data manipulation
- **Plots.jl**: Visualization

## Installation
Ensure you have Julia installed, then add the required packages:
```julia
using Pkg
Pkg.add(["JuMP", "OSQP", "MLJ", "MLJLinearModels", "DataFrames", "Plots"])
```

## Running the Project
Save the script as `portfolio_optimization.jl` and run it in Julia:
```julia
include("portfolio_optimization.jl")
```

## Project Breakdown
### 1. Mean-Variance Optimization
- Defines expected returns and covariance matrix.
- Uses JuMP.jl to solve for optimal portfolio weights.
- Constraints include weight sum = 1 and target return.
- Outputs optimal portfolio weights.

### 2. Reinforcement Learning (Q-Learning)
- Simulates a stateless 2-asset selection problem.
- Uses Q-learning to find the best asset to invest in based on reward history.
- Tracks Q-values for each asset and selects the best-performing one.

### 3. Asset Return Prediction (MLJ.jl)
- Generates synthetic data for past and market returns.
- Trains a linear regression model to predict asset returns.
- Evaluates predictions for new data points.

## Output
- Optimal portfolio weights printed and visualized in `optimal_weights.png`.
- RL reward history plotted in `rl_reward_history.png`.
- Predicted asset returns displayed in console.
