# Implementation Steps

##### Library Imports
 `numpy`, `pandas`, `torch`, `scipy.integrate.solve_bvp`, `sklearn`


##### Configuration
| Parameter | Value |
| :--- | :--- |
| Hidden Layers | 9 |
| Neurons/Layer | 30 |
| Epochs | 500 |
| Batch Size | 512 |
| Learning Rate | 0.001 |
| Train/Val/Test Split | 80/10/10 |

We created a central configuration block to hold all these variables. This let us changing the learning rate or network size in one place.

##### ODE Solver (HybridNanofluidSolver)
Solves ode equations using `scipy.solve_bvp`.

Boundary Conditions:
- f(0) = 0, f'(0) = lambda + beta*f''(0), f'(∞) = 1
- theta'(0) = -Nh(1-theta(0)), theta(∞) = 0

We programmed the main physics equations into a class that uses `solve_bvp`. This calculates the ground truth for fluid flow and temperature so the model has something accurate to learn from.

##### Data Generation (DataGenerator)
- 54 parameter combinations (M, Nr, Nh, lambda, beta)
- 400 points per case
- Output: `training_data.csv`, `test_data.csv`

We automated the solver to loop through 54 different scenarios. This generated thousands of data points, effectively creating the "textbook" for the model to learn from.

##### Preprocessing (DataPreprocessor)
- MinMaxScaler normalization [0,1]
- Data split into train/validation/test sets
- Scalers saved as `.pkl` files

We squashed all the raw numbers into a 0-1 range so the neural network could digest them easily. Then we split the data to ensure the model isn't just memorizing answers but actually learning to predict new ones.

##### ANN Architecture (ANNModel)
Input(1) → [Linear(30) + Tanh] × 9 → Output(2)
- **Input:** eta
- **Output:** f(eta), theta(eta)
- **Activation:** Tanh
- **Weight Init:** Xavier

We designed a neural network in PyTorch. We chose 9 layers with Tanh activation because fluid dynamics are smooth and continuous, and this specific structure captures those curves really well.

##### Training (Trainer)
- **Optimizer:** Adam
- **Loss:** MSE
- **Early stopping patience:** 1000 epochs

We set up a training loop where the model guesses the outcome, compares it to the real physics data, and adjusts itself. We used the Adam optimizer to make these adjustments precise and added an "early stopping" rule to save time if the model stopped improving.

##### Evaluation
- **Metrics:** MSE, MAE, Max Error (overall and per-output)

After training, we ran the model on a separate test set it had never seen before. We calculated the error rates to mathematically prove that the model's predictions are reliable.

##### Visualization
- Training/validation loss curves
- Prediction vs true value scatter plots
- Sample profile comparisons

We generated comparison graphs that overlay the AI's predictions directly onto the physics curves. This gives us instant visual proof that the two match.

##### Output Summary
Prints data statistics, model details, training results, and file paths.

We added a final summary step that prints all the key stats and file locations right to the console, giving us a quick snapshot of the experiment's success without checking every file manually.

##### Output Files
| File | Description |
| :--- | :--- |
| `data/training_data.csv` | ODE solutions (54 cases × 400 points) |
| `data/test_data.csv` | Random test cases |
| `outputs/models/best_model.pth` | Trained model checkpoint |
| `outputs/models/scaler_*.pkl` | Normalization scalers |
| `outputs/plots/*.png` | Result visualizations |
