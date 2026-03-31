# Implementation Steps

Here is the straightforward breakdown of how we built the system.

### Library Imports
**What we did:** We grabbed all the code libraries we needed to start.
**How:** We imported tools like `numpy` for math, `pandas` to manage data tables, `torch` to build the AI brain, `scipy` to solve the physics equations, and `sklearn` to organize the data.

### Configuration
**What we did:** We set up the "rules" and settings for the AI so they are easy to change later.
**How:** We created a settings block with specific values:
*   **Hidden Layers:** 9 (The depth of the AI's thinking)
*   **Neurons/Layer:** 30 (The width of each thought layer)
*   **Epochs:** 500 (How many times it practices)
*   **Batch Size:** 512 (How many problems it solves at once)
*   **Learning Rate:** 0.001 (How fast it changes its mind)
*   **Data Split:** 80% for training, 10% for validation, 10% for testing.

### ODE Solver (HybridNanofluidSolver)
**What we did:** We built a traditional physics calculator to act as the "Teacher."
**How:** We programmed the Momentum and Energy equations using `scipy.solve_bvp`. We applied specific boundary conditions (rules at the surface and far away) like `f(0) = 0` to ensure the physics were realistic.

### Data Generation (DataGenerator)
**What we did:** We created a textbook of examples for the AI to study.
**How:** We ran the physics solver on 54 different combinations of parameters (like Magnetic field `M` and Radiation `Nr`). Since each case has 400 points, we generated a massive dataset saved as `training_data.csv`.

### Preprocessing (DataPreprocessor)
**What we did:** We polished the data so the AI could understand it better.
**How:** We used a `MinMaxScaler` to instanly shrink all numbers to be between 0 and 1. We then split the data into three piles (Train, Result Check, and Final Exam) and saved the scalers so we can undo the shrinking later.

### ANN Architecture (ANNModel)
**What we did:** We constructed the actual brain of the AI.
**How:** We built a Feed-Forward network.
*   **Input:** It takes 1 number (`η`, distance).
*   **Processing:** It passes through 9 layers of 30 neurons each, using `Tanh` activation to decide what's important.
*   **Output:** It spits out 2 numbers (`f` velocity and `θ` temperature).
We used `Xavier` initialization to set the initial random weights carefully.

### Training (Trainer)
**What we did:** We taught the AI using the data.
**How:** We used the `Adam` optimizer (a smart way to correct mistakes) and measured error using `MSE` (Mean Squared Error). We set it to stop early if it didn't get better for 1000 tries to save time.

### Evaluation
**What we did:** We gave the AI a final exam.
**How:** We checked how close its answers were to the real physics answers using simple scores: `MSE` (squared error), `MAE` (absolute error), and `Max Error` (the biggest mistake it made).

### Visualization
**What we did:** We drew pictures to prove it works.
**How:** We created three main charts:
1.  **Loss Curves:** To show the error dropping over time.
2.  **Scatter Plots:** To show predictions matching real values.
3.  **Profile Comparisons:** Side-by-side lines of AI vs. Physics.

### Output Summary
**What we did:** We made the program report its own success.
**How:** At the very end, we printed out all the stats, model architecture details, and locations of the saved files so we know exactly what happened.

## Output Files

| File | Description |
| :--- | :--- |
| `data/training_data.csv` | The "textbook" (54 cases × 400 physics solutions) |
| `data/test_data.csv` | Random extra cases for testing |
| `outputs/models/best_model.pth` | The saved, trained AI brain |
| `outputs/models/scaler_*.pkl` | files to translate the normalized numbers back to real units |
| `outputs/plots/*.png` | The pictures proving the results |
