# complete ann implementation
# this file contains everything from data generation to model training and validation

# step 1: import all required libraries
print("=" * 80)
print("HYBRID NANOFLUID ANN - COMPLETE IMPLEMENTATION")
print("=" * 80)
print("\nStep 1: Importing libraries...")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_bvp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from tqdm import tqdm
import itertools
import pickle

warnings.filterwarnings('ignore')

# set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("✓ All libraries imported successfully!")


# step 2: configuration and hyperparameters
print("\nStep 2: Setting up configuration...")

class Config:
    # directory paths
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    PLOT_DIR = OUTPUT_DIR / "plots"
    
    # physics parameters for different test cases
    PARAM_RANGES = {
        'M': [0.5, 1.0, 2.0],      
        'Nr': [0.5, 1.0, 1.5],     
        'Nh': [0.1, 0.5],          
        'lam': [0.5, 1.0, 1.5],    
        'beta': [0.1, 0.2],        
        'Pr': [6.2],               
        'n': [1.0],                
        'Tr': [1.5],               
        'As': [1.0],               
    }
    
    # numerical solver settings
    ETA_MAX = 10.0        
    N_POINTS = 400        
    
    # neural network architecture
    INPUT_DIM = 1         
    HIDDEN_DIM = 30       
    NUM_HIDDEN_LAYERS = 9 
    OUTPUT_DIM = 2        
    
    # training hyperparameters
    EPOCHS = 100                    
    BATCH_SIZE = 512                
    LEARNING_RATE = 0.005           
    EARLY_STOPPING_PATIENCE = 1000  
    VAL_SPLIT = 0.1                 
    TEST_SPLIT = 0.1                

# create output directories
Config.DATA_DIR.mkdir(exist_ok=True, parents=True)
Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
Config.PLOT_DIR.mkdir(exist_ok=True, parents=True)

print(f"✓ Configuration set up!")
print(f"  - Data directory: {Config.DATA_DIR}")
print(f"  - Output directory: {Config.OUTPUT_DIR}")
print(f"  - Total parameter combinations: {np.prod([len(v) for v in Config.PARAM_RANGES.values()])}")


# step 3: physics-based ode solver for hybrid nanofluid
print("\nStep 3: Setting up physics-based ODE solver...")

class HybridNanofluidSolver:
    def __init__(self, params):
        # physics parameters
        self.M = params.get('M', 1.0)
        self.Nr = params.get('Nr', 0.5)
        self.Nh = params.get('Nh', 0.5)
        self.lam = params.get('lam', 1.0)
        self.beta = params.get('beta', 0.1)
        self.Pr = params.get('Pr', 6.2)
        self.n = params.get('n', 1.0)
        self.Tr = params.get('Tr', 1.5)
        self.As = params.get('As', 1.0)
        
        # nanofluid property ratios
        self.nu_ratio = params.get('nu_ratio', 1.05)
        self.kappa_ratio = params.get('kappa_ratio', 1.15)
        self.sigma_ratio = params.get('sigma_ratio', 1.10)
        self.rho_ratio = params.get('rho_ratio', 1.03)
        
        # solver settings
        self.eta_max = params.get('eta_max', 10.0)
        self.n_points = params.get('n_points', 400)
    
    def ode_system(self, eta, y):
        # extract variables: f, f', f'', theta, theta'
        f, fp, fpp, theta, thetap = y
        
        # momentum equation - solve for f'''
        term1 = f * fpp
        term2 = (2.0 * self.n) / (self.n + 1.0) * (1.0 - fp**2)
        term3 = -2.0 / (self.n + 1.0) * (self.sigma_ratio / self.rho_ratio * self.M) * (fp - 1.0)
        fppp = -(term1 + term2 + term3) / self.nu_ratio
        
        # energy equation - solve for theta''
        theta_term = np.maximum(1.0 + (self.Tr - 1.0) * theta, 0.01)
        coeff = self.kappa_ratio + self.Nr / (theta_term**3)
        term1_e = self.Pr * self.As * (f * thetap - 2.0 * (2.0 * self.n - 1.0) / (self.n + 1.0) * fp * theta)
        term2_e = 3.0 * self.Nr * (self.Tr - 1.0) / (theta_term**2) * (thetap**2)
        thetapp = -(term1_e + term2_e) / coeff
        
        return np.vstack([fp, fpp, fppp, thetap, thetapp])
    
    def boundary_conditions(self, ya, yb):
        # boundary conditions at eta=0 and eta=infinity
        bc = np.zeros(5)
        bc[0] = ya[0]                                   # f(0) = 0
        bc[1] = ya[1] - (self.lam + self.beta * ya[2]) # f'(0) = lam + beta*f''(0)
        bc[2] = ya[4] + self.Nh * (1.0 - ya[3])        # theta'(0) = -Nh*(1-theta(0))
        bc[3] = yb[1] - 1.0                             # f'(inf) = 1
        bc[4] = yb[3]                                   # theta(inf) = 0
        return bc
    
    def solve(self, verbose=False):
        # create grid and initial guess
        eta = np.linspace(0, self.eta_max, self.n_points)
        f_init = eta - np.exp(-eta)
        fp_init = 1.0 - np.exp(-eta)
        fpp_init = np.exp(-eta)
        theta_init = np.exp(-eta)
        thetap_init = -np.exp(-eta)
        y_init = np.vstack([f_init, fp_init, fpp_init, theta_init, thetap_init])
        
        # solve boundary value problem
        try:
            sol = solve_bvp(self.ode_system, self.boundary_conditions, eta, y_init,
                          max_nodes=5000, tol=1e-6, verbose=0)
            if sol.success:
                return sol.x, sol.y
            return None, None
        except:
            return None, None
    
    def compute_engineering_quantities(self, solution):
        # compute skin friction and nusselt number
        fpp_0 = solution[2][0]
        thetap_0 = solution[4][0]
        return {'Cf': fpp_0, 'Nu': -thetap_0, 'fpp_0': fpp_0, 'thetap_0': thetap_0}

print("✓ ODE solver implemented!")


# step 4: data generation from physics equations
print("\nStep 4: Generating training data from physics equations...")

class DataGenerator:
    def __init__(self):
        self.param_combinations = self._generate_parameter_grid()
    
    def _generate_parameter_grid(self):
        # generate all combinations of parameters
        keys = Config.PARAM_RANGES.keys()
        values = Config.PARAM_RANGES.values()
        param_combinations = []
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            params['eta_max'] = Config.ETA_MAX
            params['n_points'] = Config.N_POINTS
            param_combinations.append(params)
        
        return param_combinations
    
    def solve_single_case(self, params, case_id):
        # solve ode for one parameter set
        solver = HybridNanofluidSolver(params)
        eta, solution = solver.solve(verbose=False)
        
        if solution is None:
            return None
        
        # extract engineering quantities
        eng_quantities = solver.compute_engineering_quantities(solution)
        
        # create dataframe with results
        df = pd.DataFrame({
            'case_id': case_id,
            'eta': eta,
            'f': solution[0],
            'fp': solution[1],
            'fpp': solution[2],
            'theta': solution[3],
            'thetap': solution[4],
            'M': params['M'],
            'Nr': params['Nr'],
            'Nh': params['Nh'],
            'lam': params['lam'],
            'beta': params['beta'],
            'Pr': params['Pr'],
            'n': params['n'],
            'Tr': params['Tr'],
            'As': params['As'],
            'Cf': eng_quantities['Cf'],
            'Nu': eng_quantities['Nu']
        })
        return df
    
    def generate_dataset(self):
        # generate complete training dataset
        print(f"  Generating {len(self.param_combinations)} parameter combinations...")
        print(f"  Total data points: {len(self.param_combinations) * Config.N_POINTS}")
        
        all_data = []
        for case_id, params in enumerate(tqdm(self.param_combinations, desc="  Solving ODEs")):
            df = self.solve_single_case(params, case_id)
            if df is not None:
                all_data.append(df)
        
        # save to csv
        if all_data:
            full_dataset = pd.concat(all_data, ignore_index=True)
            output_path = Config.DATA_DIR / "training_data.csv"
            full_dataset.to_csv(output_path, index=False)
            
            print(f"\n  ✓ Dataset saved to: {output_path}")
            print(f"    - Successful cases: {len(all_data)}/{len(self.param_combinations)}")
            print(f"    - Total rows: {len(full_dataset)}")
            return full_dataset
        return None
    
    def generate_test_cases(self, n_cases=10):
        # generate random test cases
        print(f"\n  Generating {n_cases} random test cases...")
        test_data = []
        
        for case_id in tqdm(range(n_cases), desc="  Generating test cases"):
            # random parameters
            params = {
                'M': np.random.uniform(0.3, 2.5),
                'Nr': np.random.uniform(0.1, 1.2),
                'Nh': np.random.uniform(0.1, 1.2),
                'lam': np.random.uniform(0.3, 2.5),
                'beta': 0.1,
                'Pr': 6.2,
                'n': 1.0,
                'Tr': 1.5,
                'As': 1.0,
                'eta_max': Config.ETA_MAX,
                'n_points': Config.N_POINTS
            }
            
            df = self.solve_single_case(params, case_id + 1000)
            if df is not None:
                test_data.append(df)
        
        # save test data
        if test_data:
            test_dataset = pd.concat(test_data, ignore_index=True)
            output_path = Config.DATA_DIR / "test_data.csv"
            test_dataset.to_csv(output_path, index=False)
            print(f"  ✓ Test dataset saved: {len(test_dataset)} rows")
            return test_dataset
        return None

# generate training and test data
generator = DataGenerator()
train_data = generator.generate_dataset()
test_data = generator.generate_test_cases(n_cases=10)

print("\n✓ Data generation complete!")


# step 5: data preprocessing and normalization
print("\nStep 5: Preparing data for neural network training...")

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler_eta = MinMaxScaler()
        self.scaler_f = MinMaxScaler()
        self.scaler_theta = MinMaxScaler()
    
    def load_and_preprocess(self):
        # load and preprocess data
        print("  Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"  Total samples: {len(df)}")
        print(f"  Unique cases: {df['case_id'].nunique()}")
        
        # extract features and targets
        eta = df['eta'].values.reshape(-1, 1)
        f = df['f'].values.reshape(-1, 1)
        theta = df['theta'].values.reshape(-1, 1)
        
        # normalize to [0, 1]
        print("  Normalizing data...")
        eta_normalized = self.scaler_eta.fit_transform(eta)
        f_normalized = self.scaler_f.fit_transform(f)
        theta_normalized = self.scaler_theta.fit_transform(theta)
        
        # combine outputs
        targets = np.hstack([f_normalized, theta_normalized])
        
        # split into train/val/test
        print("  Splitting into train/validation/test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            eta_normalized, targets,
            test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
            random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        # convert to pytorch tensors
        return {
            'X_train': torch.FloatTensor(X_train),
            'y_train': torch.FloatTensor(y_train),
            'X_val': torch.FloatTensor(X_val),
            'y_val': torch.FloatTensor(y_val),
            'X_test': torch.FloatTensor(X_test),
            'y_test': torch.FloatTensor(y_test)
        }
    
    def save_scalers(self):
        # save scalers for later use
        with open(Config.MODEL_DIR / 'scaler_eta.pkl', 'wb') as f:
            pickle.dump(self.scaler_eta, f)
        with open(Config.MODEL_DIR / 'scaler_f.pkl', 'wb') as f:
            pickle.dump(self.scaler_f, f)
        with open(Config.MODEL_DIR / 'scaler_theta.pkl', 'wb') as f:
            pickle.dump(self.scaler_theta, f)
        print(f"  ✓ Scalers saved to {Config.MODEL_DIR}")

# preprocess data
preprocessor = DataPreprocessor(Config.DATA_DIR / "training_data.csv")
data = preprocessor.load_and_preprocess()
preprocessor.save_scalers()

print("\n✓ Data preprocessing complete!")


# step 6: neural network architecture
print("\nStep 6: Building neural network architecture...")

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(Config.INPUT_DIM, Config.HIDDEN_DIM))
        layers.append(nn.Tanh())
        
        for _ in range(Config.NUM_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(Config.HIDDEN_DIM, Config.OUTPUT_DIM))
        
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, eta):
        return self.network(eta)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_summary(self):
        # print model architecture
        summary = []
        summary.append("=" * 60)
        summary.append("ANN Architecture Summary")
        summary.append("=" * 60)
        summary.append(f"Input dimension:        {Config.INPUT_DIM}")
        summary.append(f"Hidden layers:          {Config.NUM_HIDDEN_LAYERS}")
        summary.append(f"Neurons per layer:      {Config.HIDDEN_DIM}")
        summary.append(f"Activation function:    Tanh")
        summary.append(f"Output dimension:       {Config.OUTPUT_DIM}")
        summary.append(f"Total parameters:       {self.count_parameters():,}")
        summary.append("=" * 60)
        return "\n".join(summary)

# create model
model = ANNModel()
print(model.get_summary())

print("\n✓ Neural network built successfully!")


# step 7: model training
print("\nStep 7: Training the neural network...")

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': []
        }
    
    def train_epoch(self, X_train, y_train, batch_size):
        # train for one epoch
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # shuffle data
        n_samples = len(X_train)
        indices = torch.randperm(n_samples)
        
        # process batches
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices].to(self.device)
            y_batch = y_train[batch_indices].to(self.device)
            
            # forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, X_val, y_val):
        # validate model
        self.model.eval()
        with torch.no_grad():
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            predictions = self.model(X_val)
            loss = self.criterion(predictions, y_val)
        return loss.item()
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        # complete training loop
        print(f"\n  Training Configuration:")
        print(f"  - Optimizer: Adam")
        print(f"  - Learning rate: {Config.LEARNING_RATE}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max epochs: {epochs}")
        print(f"  - Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")
        print("\n" + "=" * 70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # train and validate
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            val_loss = self.validate(X_val, y_val)
            epoch_time = time.time() - start_time
            
            # save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_time'].append(epoch_time)
            
            # print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        
        # restore best model
        self.model.load_state_dict(self.best_model_state)
        
        print("\n" + "=" * 70)
        print(f"  ✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        return self.history
    
    def evaluate(self, X_test, y_test):
        # evaluate on test set
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            predictions = self.model(X_test)
        
        # compute metrics
        mse = torch.mean((predictions - y_test) ** 2).item()
        mae = torch.mean(torch.abs(predictions - y_test)).item()
        max_error = torch.max(torch.abs(predictions - y_test)).item()
        
        # per-output metrics
        mse_f = torch.mean((predictions[:, 0] - y_test[:, 0]) ** 2).item()
        mse_theta = torch.mean((predictions[:, 1] - y_test[:, 1]) ** 2).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'mse_f': mse_f,
            'mse_theta': mse_theta
        }
    
    def save_model(self, path):
        # save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"  ✓ Model saved to {path}")

# train the model
trainer = Trainer(model)
history = trainer.train(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val'],
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE
)

print("\n✓ Training complete!")


# step 8: model evaluation
print("\nStep 8: Evaluating model performance...")

metrics = trainer.evaluate(data['X_test'], data['y_test'])

print("\n  Test Set Metrics:")
print("  " + "=" * 50)
print(f"  Overall MSE:        {metrics['mse']:.6e}")
print(f"  Overall MAE:        {metrics['mae']:.6e}")
print(f"  Max Error:          {metrics['max_error']:.6e}")
print(f"  MSE (f):            {metrics['mse_f']:.6e}")
print(f"  MSE (theta):        {metrics['mse_theta']:.6e}")
print("  " + "=" * 50)

# save model
model_path = Config.MODEL_DIR / "best_model.pth"
trainer.save_model(model_path)

print("\n✓ Evaluation complete!")


# step 9: visualization
print("\nStep 9: Creating visualizations...")

# set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# plot 1: training and validation loss
ax1 = axes[0, 0]
epochs_range = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs_range, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
ax1.plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# plot 2: time per epoch
ax2 = axes[0, 1]
ax2.plot(epochs_range, history['epoch_time'], 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Time per Epoch')
ax2.grid(True, alpha=0.3)

# plot 3: predictions vs true (f)
ax3 = axes[1, 0]
model.eval()
with torch.no_grad():
    predictions = model(data['X_test']).numpy()
y_test_np = data['y_test'].numpy()

ax3.scatter(y_test_np[:, 0], predictions[:, 0], alpha=0.5, s=10)
ax3.plot([y_test_np[:, 0].min(), y_test_np[:, 0].max()],
         [y_test_np[:, 0].min(), y_test_np[:, 0].max()], 'r--', linewidth=2)
ax3.set_xlabel('True f (normalized)')
ax3.set_ylabel('Predicted f (normalized)')
ax3.set_title('Predictions vs True Values (f)')
ax3.grid(True, alpha=0.3)

# plot 4: predictions vs true (theta)
ax4 = axes[1, 1]
ax4.scatter(y_test_np[:, 1], predictions[:, 1], alpha=0.5, s=10)
ax4.plot([y_test_np[:, 1].min(), y_test_np[:, 1].max()],
         [y_test_np[:, 1].min(), y_test_np[:, 1].max()], 'r--', linewidth=2)
ax4.set_xlabel('True θ (normalized)')
ax4.set_ylabel('Predicted θ (normalized)')
ax4.set_title('Predictions vs True Values (θ)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Config.PLOT_DIR / "training_results.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Training results plot saved to: {plot_path}")

# create sample prediction plot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# load test case
test_df = pd.read_csv(Config.DATA_DIR / "test_data.csv")
sample_case = test_df[test_df['case_id'] == 1000].copy()

# normalize and predict
eta_sample = sample_case['eta'].values.reshape(-1, 1)
eta_normalized = preprocessor.scaler_eta.transform(eta_sample)

model.eval()
with torch.no_grad():
    predictions_sample = model(torch.FloatTensor(eta_normalized)).numpy()

# denormalize predictions
f_pred = preprocessor.scaler_f.inverse_transform(predictions_sample[:, 0].reshape(-1, 1))
theta_pred = preprocessor.scaler_theta.inverse_transform(predictions_sample[:, 1].reshape(-1, 1))

# plot velocity profile
axes2[0].plot(sample_case['eta'], sample_case['f'], 'b-', linewidth=2, label='True (ODE Solution)')
axes2[0].plot(sample_case['eta'], f_pred, 'r--', linewidth=2, label='ANN Prediction')
axes2[0].set_xlabel('η')
axes2[0].set_ylabel('f(η)')
axes2[0].set_title('Velocity Profile Prediction')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# plot temperature profile
axes2[1].plot(sample_case['eta'], sample_case['theta'], 'b-', linewidth=2, label='True (ODE Solution)')
axes2[1].plot(sample_case['eta'], theta_pred, 'r--', linewidth=2, label='ANN Prediction')
axes2[1].set_xlabel('η')
axes2[1].set_ylabel('θ(η)')
axes2[1].set_title('Temperature Profile Prediction')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path2 = Config.PLOT_DIR / "sample_predictions.png"
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"  ✓ Sample predictions plot saved to: {plot_path2}")

print("\n✓ Visualization complete!")


# step 10: final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\n1. DATA GENERATION:")
print(f"   - Parameter combinations: {len(generator.param_combinations)}")
print(f"   - Total training samples: {len(train_data)}")
print(f"   - Test cases: 10")

print("\n2. MODEL ARCHITECTURE:")
print(f"   - Input: 1 neuron (η)")
print(f"   - Hidden: {Config.NUM_HIDDEN_LAYERS} layers × {Config.HIDDEN_DIM} neurons")
print(f"   - Output: 2 neurons (f, θ)")
print(f"   - Total parameters: {model.count_parameters():,}")

print("\n3. TRAINING:")
print(f"   - Optimizer: Adam")
print(f"   - Learning rate: {Config.LEARNING_RATE}")
print(f"   - Epochs completed: {len(history['train_loss'])}")
print(f"   - Best validation loss: {min(history['val_loss']):.6e}")
print(f"   - Total training time: {sum(history['epoch_time']):.2f} seconds")

print("\n4. PERFORMANCE:")
print(f"   - Test MSE: {metrics['mse']:.6e}")
print(f"   - Test MAE: {metrics['mae']:.6e}")
print(f"   - Max Error: {metrics['max_error']:.6e}")

print("\n5. OUTPUT FILES:")
print(f"   - Model: {model_path}")
print(f"   - Scalers: {Config.MODEL_DIR}")
print(f"   - Plots: {Config.PLOT_DIR}")
print(f"   - Data: {Config.DATA_DIR}")

print("\n" + "=" * 80)
print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 80)
