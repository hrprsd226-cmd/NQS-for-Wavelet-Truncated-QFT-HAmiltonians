import numpy as np
import pywt
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from functools import lru_cache
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
# Importing functional.hessian and jacobian for now, but will show an optimized way
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='pywt')

# Suppress warnings from PyWavelets if desired (e.g., for missing wavelets)
warnings.filterwarnings("ignore", category=UserWarning, module='pywt')

# --- Configuration --- #
K_RESOLUTION = 0
L_DOMAIN = 15
E_TARGET = 11.6511 # set exact energy here
EPSILON = 1e-1              # desired precision
PARTICLE_MASS = 1.0
WAVELET_TYPE = 'db4'
WAVELET_INTERNAL_RESOLUTION = 10
DTYPE = torch.float32 # Define a global dtype for consistency

# --- Parallelism & Hyperparameters ---
NUM_PROCESSES = os.cpu_count() or 4 # Use all available cores, or a default
N_NEURONS = 200
LEARNING_RATE = 1e-4  # Slightly adjusted learning rate often works better with Adam and larger batches/vectorization
N_EPOCHS = 40000       # Max epochs; will stop early if converged
N_SAMPLES = 30000     # Increased samples for better statistics
N_EQUIL_STEPS = 5000
MC_STEP_SIZE = 0.2  # Slightly adjusted step size can improve acceptance
GRAD_CLIP_MAX_NORM = 1.0
### --- Early Stopping Parameters ---
STOPPING_PATIENCE = 50    # Number of epochs to average over for stopping
STOPPING_THRESHOLD = 5e-4 # Stop when relative stdev is less than 0.05%

os.environ['OMP_NUM_THREADS'] = str(NUM_PROCESSES)
os.environ['MKL_NUM_THREADS'] = str(NUM_PROCESSES)
torch.set_num_threads(NUM_PROCESSES)
# --- Weigt Initiation---
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
# --- Hamiltonian Setup ---

# Memoization for compute_D_matrix_element can be very beneficial
@lru_cache(maxsize=None)
def compute_D_matrix_element(m, n, k, domain_end, step_size):
    # Ensure inputs are hashable for lru_cache if they are not primitives
    grid, phi_m = get_interpolated_basis(m, k, domain_end, step_size)
    _, phi_n = get_interpolated_basis(n, k, domain_end, step_size)
    d_phi_m = np.gradient(phi_m, grid)
    d_phi_n = np.gradient(phi_n, grid)
    return trapezoid(d_phi_m * d_phi_n, grid)

@lru_cache(maxsize=None)
def get_wavelet_basis(n, k, level=WAVELET_INTERNAL_RESOLUTION):
    wavelet = pywt.Wavelet(WAVELET_TYPE)
    # Using 'db' wavelets, 'phi' is the scaling function, 'psi' is the wavelet function
    # For a basis, we usually use the scaling functions.
    # Note: pywt.wavefun only returns scaling function (phi) and wavelet (psi) for certain levels.
    # If level is too high, it might return None for some components.
    phi, _, x = wavelet.wavefun(level=level)
    if phi is None: # Fallback if specific level doesn't provide phi
         phi, _, x = wavelet.wavefun(level=pywt.dwt_max_level(len(wavelet.dec_lo) * (2**level - 1), wavelet.dec_lo))
    
    # Scale and shift the basis functions
    x_scaled = (x + n) / (2 ** k)
    phi_scaled = 2 ** (k / 2) * phi
    return x_scaled, phi_scaled

@lru_cache(maxsize=None)
def get_interpolated_basis(n, k, domain_end, step_size):
    x_db, y_db = get_wavelet_basis(n, k)
    x_full = np.arange(0, domain_end, step_size)
    # Use np.interp for more robust interpolation
    y_full = np.interp(x_full, x_db, y_db, left=0, right=0)
    return x_full, y_full

def get_hamiltonian_matrices(k, L, m):
    # N is the number of basis functions. The original calculation might be off for some wavelets.
    # A more robust way is to determine N based on the domain and wavelet support.
    # Here, we keep your original N calculation but add a check.
    N_approx = int(L * (2**k)) # Approximate number of basis functions that fit within L
    
    # Determine the effective extent of the mother wavelet (phi_0,0)
    mother_wavelet_x, _ = get_wavelet_basis(0,0)
    # Handle cases where wavelet support is very small or extends beyond the first few grid points
    if len(mother_wavelet_x) < 2:
        raise ValueError("Mother wavelet has insufficient points for width calculation.")
    
    mother_wavelet_width = mother_wavelet_x[-1] - mother_wavelet_x[0]

    # Calculate N based on L_DOMAIN and wavelet width.
    # This determines how many basis functions 'fit' in the domain L.
    # The -4 from your original code is heuristic, often related to boundary effects or specific wavelet choices.
    # Let's use a more direct approach: we need N basis functions whose centers are roughly within [0, L].
    # The actual support of each basis function (n, k) is [n*2^-k, (n+width)*2^-k].
    # We want basis functions whose support significantly overlaps with [0, L].
    # For simplicity, we'll keep your N calculation based on L * (2**k) - 4 but note it's an approximation.
    N = int(L * (2**k)) - 4
    
    if N <= 1:
        raise ValueError(f"k={k} and L={L} results in N={N}, which is too small for a meaningful simulation. "
                         "Consider increasing L or k.")
    print(f"Hamiltonian initialized for k={k} with N={N} degrees of freedom.")
    
    # Adjust domain_end calculation: It should define the full range over which basis functions are defined.
    # A safer approach is to define it once and use it consistently.
    # If the basis functions are centered, a range like [0, L] is usually sufficient.
    # For wavelets, the exact 'domain_end' for interpolation needs to cover the support of all N basis functions.
    # Let's set domain_end as L for field expectation plotting, assuming our N basis functions cover it.
    
    # Calculate step_size from the reference wavelet
    ref_x, _ = get_wavelet_basis(0, k)
    if len(ref_x) < 2:
        raise ValueError("Reference wavelet has insufficient points to determine step_size.")
    step_size = ref_x[1] - ref_x[0]
    
    # Pre-allocate D_matrix and ensure it's a float type
    D_matrix = np.zeros((N, N), dtype=np.float32)
    
    print(f"Calculating D_matrix ({N}x{N})... This might take a while.")
    for r in tqdm(range(N), desc="Calculating D_matrix rows"):
        for c in range(r, N): # Utilize symmetry
            val = compute_D_matrix_element(r, c, k, L, step_size) # Use L as domain_end
            D_matrix[r, c] = val
            D_matrix[c, r] = val
            
    # Return L as domain_end for consistency with post-processing
    return torch.from_numpy(D_matrix).to(DTYPE), torch.tensor(m, dtype=DTYPE), N, L, step_size

# --- Neural Network Wavefunction Ansatz ---

class NQS_Ansatz(nn.Module):
    def __init__(self, N, n_neurons):
        super(NQS_Ansatz, self).__init__()
        self.N = N
        self.nn = nn.Sequential(
            nn.Linear(N, n_neurons),
            nn.Softplus(), # Using a smooth activation
            nn.Linear(n_neurons, n_neurons),
            nn.Softplus(),
            #nn.Linear(n_neurons, n_neurons),
            #nn.Softplus(),
            nn.Linear(n_neurons, 1)
        )

    def forward(self, phi):
        # The output is log(|Psi|) for VMC, so it's a scalar for each configuration.
        # Ensure phi is correctly shaped (batch_size, N)
        return self.nn(phi).squeeze(-1) # Squeeze to make it (batch_size,)
#def init_weights(m):
 #   if isinstance(m, nn.Linear):
  #      # Change gain=0.1 to gain=0.01 or use a standard Kaiming/Xavier
        # Lower gain prevents large gradients during early epochs.
   #     torch.nn.init.xavier_uniform_(m.weight, gain=0.01) 
    #    if m.bias is not None:
     #       m.bias.data.fill_(0.0) # Start bias near zero
# --- Variational Monte Carlo Components ---

def metropolis_hastings_sampler(model, n_samples, n_equil_steps, step_size, initial_state, device):
    """
    Modified to calculate and return the acceptance rate.
    Ensures all tensors are on the correct device.
    """
    current_phi = initial_state.to(device)
    
    # It's crucial that the model is in eval mode during sampling if it has dropout/batchnorm,
    # though NQS_Ansatz typically doesn't have these.
    # It's also important to use torch.no_grad() for sampling phase.
    with torch.no_grad():
        log_psi_current = model(current_phi)
        
    samples = torch.zeros((n_samples, initial_state.shape[1]), device=device, dtype=DTYPE)
    
    accepted_count = 0
    total_proposals = 0
    
    # Pre-generate random numbers for efficiency
    random_normals = torch.randn(n_samples + n_equil_steps, current_phi.shape[1], device=device, dtype=DTYPE)
    random_uniforms = torch.rand(n_samples + n_equil_steps, device=device, dtype=DTYPE)

    for i in range(n_samples + n_equil_steps):
        # Use pre-generated random normal
        proposed_phi = current_phi + random_normals[i].unsqueeze(0) * step_size
        
        with torch.no_grad():
            log_psi_proposed = model(proposed_phi)
            
        log_acceptance_ratio = 2 * (log_psi_proposed - log_psi_current)
        
        # Use pre-generated random uniform
        accept_mask = (torch.log(random_uniforms[i]) < log_acceptance_ratio).squeeze()
        
        if accept_mask:
            current_phi = proposed_phi
            log_psi_current = log_psi_proposed
            accepted = True
        else:
            accepted = False
            
        if i >= n_equil_steps:
            samples[i - n_equil_steps] = current_phi.squeeze(0) # Store the un-squeezed vector
            total_proposals += 1
            if accepted:
                accepted_count += 1
    
    acc_rate = accepted_count / total_proposals if total_proposals > 0 else 0.0
    if i % (n_samples // 10) == 0: # Print every 10% of samples
        print(f"  Sample {i}: log_psi_current={log_psi_current.item():.4f}, log_psi_proposed={log_psi_proposed.item():.4f}, log_ratio={log_acceptance_ratio.item():.4f}, rand_unif_log={torch.log(random_uniforms[i]).item():.4f}")
    return samples, acc_rate

# OPTIMIZED calculate_local_energy
def calculate_local_energy(model, phi, D_matrix, m_particle):
    # If using DataParallel, unwrap the model to get the underlying module
    underlying_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Set requires_grad=True for phi to enable gradient computations
    phi_requires_grad = phi.clone().detach().requires_grad_(True)
    
    # Compute log_psi for all samples
    log_psi_batch = underlying_model(phi_requires_grad) # Shape: (batch_size,)

    # Potential Energy
    # v_mass: 0.5 * m^2 * sum(phi_i^2) for each sample
    # v_coupling: 0.5 * sum_ij (phi_i * D_ij * phi_j) for each sample
    # Both can be computed efficiently for the whole batch
    
    # Mass term: 0.5 * m_particle^2 * ||phi||^2
    # Reshape phi_requires_grad for element-wise multiplication
    v_mass = 0.5 * m_particle**2 * torch.sum(phi_requires_grad**2, dim=1) # Shape: (batch_size,)

    # Coupling term: 0.5 * phi @ D_matrix @ phi.T
    # torch.einsum('bi,ij,bj->b', phi, D_matrix, phi) is correct and efficient.
    v_coupling = 0.5 * torch.einsum('bi,ij,bj->b', phi_requires_grad, D_matrix, phi_requires_grad) # Shape: (batch_size,)
    
    potential_energy = v_mass + v_coupling # Shape: (batch_size,)

    # Kinetic Energy (Laplacian term)
    # We need -0.5 * (Laplacian(log_psi) + ||grad(log_psi)||^2)
    
    # 1. Compute gradients of log_psi with respect to phi (jacobian)
    # This will be a (batch_size, N) tensor
    grad_log_psi_batch = torch.autograd.grad(
        outputs=log_psi_batch,
        inputs=phi_requires_grad,
        grad_outputs=torch.ones_like(log_psi_batch), # Dummy grad_outputs for scalar-output function
        create_graph=True, # Need to create graph for second derivatives
        retain_graph=True  # Retain graph for future calls if needed
    )[0] # [0] because grad returns a tuple

    # ||grad(log_psi)||^2 term
    grad_log_psi_sum_sq = torch.sum(grad_log_psi_batch**2, dim=1) # Shape: (batch_size,)

    # 2. Compute Laplacian(log_psi) = sum_j (d^2 log_psi / d phi_j^2)
    # This is the trace of the Hessian.
    # We can compute this more efficiently by summing the gradients of grad_log_psi_batch
    # with respect to phi_requires_grad, specifically the diagonal elements.
    
    laplacian_log_psi_batch = torch.zeros(phi.shape[0], device=phi.device, dtype=DTYPE)

    # Manual loop for diagonal Hessian terms (most efficient if vmap for hessian is not direct)
    # This loop is over N, the dimension of phi, not the batch size.
    # It's more efficient than looping over batch_size and calling hessian for each.
    for i in range(phi_requires_grad.shape[1]): # Iterate over the N dimensions
        # Compute the gradient of the i-th component of grad_log_psi_batch w.r.t. phi_requires_grad
        # This gives us d(d(log_psi)/d(phi_i))/d(phi_j)
        grad_of_grad_i = torch.autograd.grad(
            outputs=grad_log_psi_batch[:, i], # Select the i-th component for all samples in the batch
            inputs=phi_requires_grad,
            grad_outputs=torch.ones_like(grad_log_psi_batch[:, i]),
            retain_graph=True, # Keep graph for next iteration
            create_graph=False # No need for higher order derivatives here
        )[0]
        # The i-th diagonal element of the Hessian is the i-th component of grad_of_grad_i
        # which corresponds to d^2(log_psi)/d(phi_i)^2
        laplacian_log_psi_batch += grad_of_grad_i[:, i] # Add the diagonal component for each sample

    kinetic_energies = -0.5 * (laplacian_log_psi_batch + grad_log_psi_sum_sq)
    
    # Detach to ensure these do not affect gradients of model parameters during optimization
    return (kinetic_energies + potential_energy).detach()
#--To calculate the auto-correlation time--
def calculate_integrated_autocorrelation_time(series):
    """
    Calculates the integrated autocorrelation time (tau) for a time series.
    Uses a windowing heuristic to truncate the summation when noise dominates.
    """
    series = np.array(series)
    n = len(series)
    if n == 0: return 0
    
    mean = np.mean(series)
    # Variance (C_0)
    c0 = np.var(series)
    if c0 == 0: return 1.0

    tau = 1.0
    # Iterate through lags
    for t in range(1, n):
        # Calculate auto-covariance at lag t
        ct = np.mean((series[:-t] - mean) * (series[t:] - mean))
        rho = ct / c0
        
        # Stop if correlation drops below zero (heuristic for finite series)
        if rho <= 0:
            break
        tau += 2 * rho
        
    return tau
#%%
if __name__ == '__main__':
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Setup System
    D_matrix, m_particle, N, domain_end, step_size = get_hamiltonian_matrices(K_RESOLUTION, L_DOMAIN, PARTICLE_MASS)
    
    # 2. Setup Model & Optimizer
    model = NQS_Ansatz(N, N_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Move constants to device
    D_matrix = D_matrix.to(device=device, dtype=DTYPE)
    m_particle = m_particle.to(device=device, dtype=DTYPE)

    # 3. Training Initialization
    initial_state = torch.randn(1, N, device=device, dtype=DTYPE)
    
    # Metric Trackers
    
    gradient_norm_history = [] # New: Track gradient norms
    energy_rolling_history = deque(maxlen=STOPPING_PATIENCE)
    
    min_energy_observed = float('inf')
    
    # --- START TRAINING TIMER ---
    start_time = time.time()
    print("\nStarting VMC training...")
    training_start_time = time.time()
    energy_history = []
    for epoch in tqdm(range(N_EPOCHS), desc="VMC Training"):
        epoch_start_time = time.time()
        
        # A. Sampling
        samples, acc_rate = metropolis_hastings_sampler(model, N_SAMPLES, N_EQUIL_STEPS, MC_STEP_SIZE, initial_state.detach(), device)
        initial_state = samples[-1].unsqueeze(0).detach()

        # B. Energy Calculation
        local_energies = calculate_local_energy(model, samples, D_matrix, m_particle)
        E_expectation = torch.mean(local_energies)
        
        # C. Loss & Backprop
        log_psi_samples = model(samples)
        loss = torch.mean((local_energies - E_expectation.detach()) * 2 * log_psi_samples)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nUnstable loss at epoch {epoch}. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        
        # D. Gradient Clipping & Norm Tracking
        # clip_grad_norm_ returns the total norm of the parameters *before* clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        gradient_norm_history.append(total_norm.item())
        
        optimizer.step()

        # E. History & Early Stopping
        current_energy = E_expectation.item()
        energy_history.append(current_energy)
        energy_rolling_history.append(current_energy)
        # --- T_epsilon recording ---
        elapsed_time = time.time() - training_start_time

        
        if len(energy_rolling_history) == STOPPING_PATIENCE:
            rolling_mean = np.mean(energy_rolling_history)
            rolling_std = np.std(energy_rolling_history)
            relative_std = rolling_std / abs(rolling_mean) if rolling_mean != 0 else 0
            
            if relative_std < STOPPING_THRESHOLD:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. RelStd: {relative_std:.4e}")
                break
        
        #if (epoch + 1) % 100 == 0:
            #tqdm.write(f'Epoch {epoch+1}, E: {current_energy:.5f}, Acc: {acc_rate:.2f}, GradNorm: {total_norm:.2f}')

    # --- END TRAINING TIMER ---
    end_time = time.time()
    training_wall_time = end_time - start_time
    print(f"\nTraining finished in {training_wall_time:.2f} seconds.")

    # --- POST-TRAINING ANALYSIS & METRICS ---
    print("\n--- Calculating Final Metrics ---")
    
    # 1. Generate high-quality final samples
    final_n_samples = 50000 
    with torch.no_grad():
        final_samples, final_acc = metropolis_hastings_sampler(
            model, final_n_samples, n_equil_steps=10000, 
            step_size=MC_STEP_SIZE, initial_state=initial_state.detach(), device=device
        )
        # Note: final_samples is detached from any previous graph by the sampler.
        # It is ready to be used as an input to calculate_local_energy outside this block.
    
    # 2. Calculate local energies for every sample for statistics (MOVE THIS OUTSIDE NO_GRAD)
    # We need the gradient graph for the kinetic term calculation.
    final_local_energies_tensor = calculate_local_energy(model, final_samples, D_matrix, m_particle)
    final_local_energies = final_local_energies_tensor.cpu().numpy()
    
    # 3. Calculate Metrics
    E_0_final = np.mean(final_local_energies)
    sigma_sq = np.var(final_local_energies)
    
    # Metric: Relative Variance (σ²/E₀) (Note: sometimes defined as σ²/E₀^2, check your specific need)
    # Here we compute σ² / |E₀|
    relative_variance = sigma_sq / abs(E_0_final)
    
    # Metric: Autocorrelation Time (tau)
    # We calculate this on the energy time series
    tau = calculate_integrated_autocorrelation_time(final_local_energies)
    
    print(f"Final Energy (E_0): {E_0_final:.6f}")
    print(f"Energy Variance (σ²): {sigma_sq:.6f}")
    print(f"Relative Variance (σ²/|E_0|): {relative_variance:.6f}")
    print(f"Autocorrelation Time (τ): {tau:.2f} steps")
    print(f"Effective Sample Size (N_eff): {final_n_samples / tau:.1f}")
    print(f"Training Wall Time: {training_wall_time:.2f} s")
    
    # Create a dictionary containing everything needed to reload or analyze
    checkpoint = {
        # Architecture Config
        'config': {
            'N_basis': N,
            'n_neurons': N_NEURONS,
            'k_res': K_RESOLUTION,
            'L_domain': L_DOMAIN,
        },
        # Model Weights
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Calculated Metrics
        'metrics': {
            'final_energy': E_0_final,
            'variance': sigma_sq,
            'relative_variance': relative_variance,
            'autocorrelation_time': tau,
            'training_time_seconds': training_wall_time,
            'energy_history': energy_history,
            'gradient_norm_history': gradient_norm_history
        }
    
    }


#%%
exact_energy = np.zeros(len(energy_history))
epoch = np.arange(1,len(energy_history)+1,1)
for i in range(len(exact_energy)):
    exact_energy[i] =11.6511
plt.figure(figsize=(17, 9))
plt.plot(epoch[:],energy_history[:],label='NQFS Ground state Energy',c='r',ls="-")
plt.plot(epoch[:],exact_energy[:], ls='-.',c='k',label='Exact Ground state Energy =11.6511')
plt.xlabel('Epoch',fontsize=23)
plt.ylabel('Ground State Energy $\langle H \\rangle$',fontsize=23)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.yscale('log') 
#plt.title('VMC Ground State Energy Convergence')
plt.grid(True)
plt.legend(fontsize=27)
plt.show()