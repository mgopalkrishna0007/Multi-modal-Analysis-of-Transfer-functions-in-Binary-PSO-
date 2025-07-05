import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
from scipy.special import erf, expit
from datetime import datetime
from scipy.stats import ranksums

# Create main output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_output_dir = f"BPSO_Comparison_Results_{timestamp}"
os.makedirs(main_output_dir, exist_ok=True)

# Create subdirectories
convergence_dir = os.path.join(main_output_dir, "1_Convergence_Plots")
results_dir = os.path.join(main_output_dir, "2_Results_Data")
diversity_dir = os.path.join(main_output_dir, "3_Diversity_Plots")
heatmap_dir = os.path.join(main_output_dir, "4_Heatmaps")
table_dir = os.path.join(main_output_dir, "5_Summary_Tables")

for directory in [convergence_dir, results_dir, diversity_dir, heatmap_dir, table_dir]:
    os.makedirs(directory, exist_ok=True)

# Binary encoding parameters
BITS_PER_VAR = 15  # 1 sign bit + 14 magnitude bits
SIGN_BIT = 1
MAGNITUDE_BITS = 14

def real_to_binary(x, search_range):
    """Convert real value to binary string"""
    normalized = (x - search_range[0]) / (search_range[1] - search_range[0])
    int_val = int(normalized * (2**MAGNITUDE_BITS - 1))
    binary_str = format(int_val, f'0{MAGNITUDE_BITS}b')
    sign_bit = '0' if x >= 0 else '1'
    return sign_bit + binary_str

def binary_to_real(binary_str, search_range):
    """Convert binary string to real value"""
    sign_bit = binary_str[0]
    magnitude_str = binary_str[1:]
    int_val = int(magnitude_str, 2)
    normalized = int_val / (2**MAGNITUDE_BITS - 1)
    real_val = search_range[0] + normalized * (search_range[1] - search_range[0])
    return -real_val if sign_bit == '1' else real_val

# Benchmark functions (subset for demonstration)
benchmark_functions = [
    {"name": "F1: Shifted sphere", 
     "func": lambda x: np.sum((x - 50)**2) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F2: Shifted Schwefel", 
     "func": lambda x: np.sum((x - 50)**2) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F3: Shifted rotated high conditioned elliptic", 
     "func": lambda x: np.sum((x - 50)**2) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F4: Shifted Schwefel with noise in fitness", 
     "func": lambda x: np.sum((x - 50)**2) + 450 * (1 + 0.4 * np.abs(np.random.normal(0, 1))),
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F5: Schwefel's problem 2.6 with global optimum on bounds", 
     "func": lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x)),
     "range": [-10, 10], "dim": 5, "fmin": 0},
    
    {"name": "F6: Shifted Rosenbrock", 
     "func": lambda x: np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2) + 390,
     "range": [-100, 100], "dim": 5, "fmin": 390},
    
    {"name": "F7: Shifted rotated Griewank without bounds", 
     "func": lambda x: 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))),
     "range": [-600, 600], "dim": 5, "fmin": 0},
    
    {"name": "F8: Shifted rotated Ackley with global optimum on bounds", 
     "func": lambda x: -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/len(x))) - np.exp(np.sum(np.cos(2*np.pi*x)/len(x)) + 20 + np.exp(1)),
     "range": [-32, 32], "dim": 5, "fmin": 0},
    
    {"name": "F9: Shifted Rastrigin", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 330,
     "range": [-5, 5], "dim": 5, "fmin": 330},
    
    {"name": "F10: Shifted rotated Rastrigin", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 330,
     "range": [-5, 5], "dim": 5, "fmin": 330},
    
    {"name": "F11: Shifted rotated Weierstrass", 
     "func": lambda x: np.sum([(0.5**k)*np.cos(2*np.pi*(3**k)*(x+0.5)) for k in range(0, 12)]) - len(x)*np.sum([(0.5**k)*np.cos(2*np.pi*(3**k)*0.5) for k in range(0, 12)]) + 90,
     "range": [-0.5, 0.5], "dim": 5, "fmin": 90},
    
    {"name": "F12: Schwefel's problem 2.13", 
     "func": lambda x: np.sum(x**2) + (np.sum(0.5*x))**2 + (np.sum(0.5*x))**4,
     "range": [-100, 100], "dim": 5, "fmin": 0},
    
    {"name": "F13: Expanded extended Griewank plus Rosenbrock", 
     "func": lambda x: (1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))) + np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))),
     "range": [-3, 1], "dim": 5, "fmin": 0},
        
    {"name": "F14: Shifted rotated expanded Scaffer's F6", 
    "func": lambda x: np.sum(
        0.5 + (np.sin(np.sqrt(np.maximum(x[:-1]**2 + x[1:]**2, 1e-10)))**2 - 0.5) / 
        (1 + 0.001*np.maximum(x[:-1]**2 + x[1:]**2, 1e-10))**2
    ),
    "range": [-100, 100], "dim": 5, "fmin": 0},
    
    {"name": "F15: Hybrid composition function", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 120,
     "range": [-5, 5], "dim": 5, "fmin": 120},
    
    {"name": "F16: Rotated hybrid composition function", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 10,
     "range": [-5, 5], "dim": 5, "fmin": 10},
    
    {"name": "F17: Rotated hybrid composition with narrow basin", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 10,
     "range": [-5, 5], "dim": 5, "fmin": 10},
    
    {"name": "F18: Rotated hybrid composition function", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 10,
     "range": [-5, 5], "dim": 10, "fmin": 10},
    
    {"name": "F19: Rotated hybrid composition with narrow basin", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 10,
     "range": [-5, 5], "dim": 10, "fmin": 10},
    
    {"name": "F20: Rotated hybrid composition with global opt on bounds", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 10,
     "range": [-5, 5], "dim": 10, "fmin": 10},
    
    {"name": "F21: Rotated hybrid composition function", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 360,
     "range": [-5, 5], "dim": 10, "fmin": 360},
    
    {"name": "F22: Rotated hybrid composition with high condition number matrix", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 360,
     "range": [-5, 5], "dim": 10, "fmin": 360},
    
    {"name": "F23: Non-continuous rotated hybrid composition", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 360,
     "range": [-5, 5], "dim": 10, "fmin": 360},
    
    {"name": "F24: Rotated hybrid composition function", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 260,
     "range": [-5, 5], "dim": 10, "fmin": 260},
    
    {"name": "F25: Rotated hybrid composition without bounds", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 260,
     "range": [-2, 5], "dim": 10, "fmin": 260}
]

# Transfer functions for the algorithms
transfer_functions = {
    "v4": {"name": "V4", "func": lambda v: np.abs((2/np.pi)*(np.arctan((np.pi/2)*v))), "type": "v"},
    "v5": {"name": "V5", "func": lambda v: (2/np.pi)*(np.arctan(v**2)), "type": "v"},
    "v6": {"name": "V6", "func": lambda v: 1-np.exp((-1/2)*(np.abs(v))), "type": "v"},
}

# Diversity control algorithms
def diversity_reference_algo1(t, Tmax, a=1.0, b=0.5):
    """Diversity algo1: linear decrease then 0 with a and b parameters"""
    if t < b * Tmax:
        return a * (1 - t / (b * Tmax))
    else:
        return 0

def diversity_reference_algo2(t, Tmax, a=1.0, b=0.5):
    """Diversity algo2: max diversity for first half, then 0"""
    if t < 0.5 * Tmax:
        return a
    else:
        return 0

def diversity_reference_algo3(t, Tmax, a=1.0, b=0.5):
    """Diversity algo3: sigmoid decrease from max to 0"""
    k = 10  # Steepness of sigmoid
    midpoint = b * Tmax
    return a * (1 - expit(k * (t - midpoint) / Tmax))

def initialize_particles(func_info, num_particles):
    """Initialize particles with binary representation"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    
    # Initialize real-valued positions
    real_positions = np.random.uniform(search_range[0], search_range[1], (num_particles, dim))
    
    # Convert to binary
    binary_particles = np.zeros((num_particles, particle_dim), dtype=int)
    for i in range(num_particles):
        binary_str = ''
        for x in real_positions[i]:
            binary_str += real_to_binary(x, search_range)
        binary_particles[i] = [int(b) for b in binary_str]
    
    return binary_particles

def evaluate_particle(binary_particle, func_info):
    """Convert binary to real and evaluate"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    binary_str = ''.join(map(str, binary_particle))
    
    # Split into variables
    real_position = []
    for i in range(dim):
        start = i * BITS_PER_VAR
        end = start + BITS_PER_VAR
        var_bits = binary_str[start:end]
        real_position.append(binary_to_real(var_bits, search_range))
    
    return func_info["func"](np.array(real_position))

def calculate_diversity(particles, method='hamming'):
    """Calculate population diversity using specified method"""
    num_particles, particle_dim = particles.shape
    
    if method == 'hamming':
        # Hamming distance method
        total_distance = 0
        count = 0
        for i in range(num_particles):
            for j in range(i+1, num_particles):
                total_distance += np.sum(particles[i] != particles[j])
                count += 1
        return total_distance / count if count > 0 else 0
    else:
        # Mean absolute deviation method (1/nD sum(sum(abs(x - x_bar))))
        mean_particle = np.mean(particles, axis=0)
        abs_deviations = np.abs(particles - mean_particle)
        return np.sum(abs_deviations) / (num_particles * particle_dim)

def algo1_binary_pso(func_info, num_particles=30, max_iter=500, runs=20, 
                    tf_name="algo1", div_algo="algo1", div_params={'a':1.0, 'b':0.5}):
    """Algorithm 1 implementation with fixed parameters"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    c1, c2 = 1.49, 1.49  # Fixed values for algo1
    w_initial, w_final = 1.0, 0.1
    
    all_best_fitness = np.zeros((runs, max_iter))
    all_final_fitness = np.zeros(runs)
    execution_times = np.zeros(runs)
    all_diversity = np.zeros((runs, max_iter))
    
    for run in range(runs):
        start_time = time.time()
        
        # Initialize particles with validation
        particles = initialize_particles(func_info, num_particles)
        velocities = np.random.uniform(-6, 6, (num_particles, particle_dim))
        
        # Initialize fitness with validation
        current_fitness = np.zeros(num_particles)
        for i in range(num_particles):
            try:
                current_fitness[i] = evaluate_particle(particles[i], func_info)
            except:
                current_fitness[i] = float('inf')  # Assign worst possible fitness if evaluation fails
        
        personal_best_positions = particles.copy()
        personal_best_scores = current_fitness.copy()
        
        # Find global best with validation
        valid_indices = np.where(personal_best_scores != float('inf'))[0]
        if len(valid_indices) == 0:
            global_best_idx = 0
            global_best_score = float('inf')
        else:
            global_best_idx = valid_indices[np.argmin(personal_best_scores[valid_indices])]
            global_best_score = personal_best_scores[global_best_idx]
        
        global_best_position = personal_best_positions[global_best_idx].copy()
        
        for iter in range(max_iter):
            w = w_initial - (w_initial - w_final) * iter / max_iter  # Decreasing inertia
            
            for i in range(num_particles):
                r1, r2 = np.random.random(particle_dim), np.random.random(particle_dim)
                velocities[i] = w * velocities[i] + \
                              c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                              c2 * r2 * (global_best_position - particles[i])
                
                # Transfer function with bounds checking
                try:
                    tf_values = transfer_functions[tf_name]["func"](velocities[i])
                except:
                    tf_values = np.zeros(particle_dim)
                
                # Flip-based strategy with validation
                particles[i] = np.where(np.random.random(particle_dim) < tf_values,
                                      1 - particles[i],  # Flip bit
                                      particles[i])
                
                # Evaluate with error handling
                try:
                    current_fitness[i] = evaluate_particle(particles[i], func_info)
                except:
                    current_fitness[i] = float('inf')
                
                # Update personal best if valid
                if current_fitness[i] < personal_best_scores[i] and current_fitness[i] != float('inf'):
                    personal_best_scores[i] = current_fitness[i]
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best if valid
                    if current_fitness[i] < global_best_score:
                        global_best_score = current_fitness[i]
                        global_best_position = particles[i].copy()
                        global_best_idx = i
            
            all_best_fitness[run, iter] = global_best_score
            all_diversity[run, iter] = calculate_diversity(particles)
        
        all_final_fitness[run] = global_best_score
        execution_times[run] = time.time() - start_time
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    avg_time = np.mean(execution_times)
    avg_diversity = np.mean(all_diversity, axis=0)
    
    return avg_best_fitness, avg_final_fitness, std_final_fitness, avg_time, avg_diversity

def algo2_binary_pso(func_info, num_particles=30, max_iter=500, runs=20, 
                    tf_name="algo2", div_algo="algo1", div_params={'a':1.0, 'b':0.5}):
    """Algorithm 2 implementation with adaptive parameters based on diversity"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    w_initial, w_final = 1.0, 0.1
    c1_min, c1_max = 1.0, 2.0
    c2_min, c2_max = 1.0, 2.0
    
    all_best_fitness = np.zeros((runs, max_iter))
    all_final_fitness = np.zeros(runs)
    execution_times = np.zeros(runs)
    all_diversity = np.zeros((runs, max_iter))
    all_c1_values = np.zeros((runs, max_iter))
    all_c2_values = np.zeros((runs, max_iter))
    
    # Select diversity reference function
    if div_algo == "algo1":
        div_func = diversity_reference_algo1
    elif div_algo == "algo2":
        div_func = diversity_reference_algo2
    elif div_algo == "algo3":
        div_func = diversity_reference_algo3
    
    for run in range(runs):
        start_time = time.time()
        
        # Initialize particles with validation
        particles = initialize_particles(func_info, num_particles)
        velocities = np.random.uniform(-6, 6, (num_particles, particle_dim))
        
        # Initialize fitness with validation
        current_fitness = np.zeros(num_particles)
        for i in range(num_particles):
            try:
                current_fitness[i] = evaluate_particle(particles[i], func_info)
            except:
                current_fitness[i] = float('inf')  # Assign worst possible fitness if evaluation fails
        
        personal_best_positions = particles.copy()
        personal_best_scores = current_fitness.copy()
        
        # Find global best with validation
        valid_indices = np.where(personal_best_scores != float('inf'))[0]
        if len(valid_indices) == 0:
            global_best_idx = 0
            global_best_score = float('inf')
        else:
            global_best_idx = valid_indices[np.argmin(personal_best_scores[valid_indices])]
            global_best_score = personal_best_scores[global_best_idx]
        
        global_best_position = personal_best_positions[global_best_idx].copy()
        
        for iter in range(max_iter):
            w = w_initial - (w_initial - w_final) * iter / max_iter  # Decreasing inertia
            
            # Calculate current diversity D0(t)
            try:
                D0 = calculate_diversity(particles)
            except:
                D0 = 0.0
            
            # Calculate reference diversity Di(t)
            try:
                Di = div_func(iter, max_iter, **div_params)
                e = Di - D0
            except:
                e = 0.0
            
            # Adaptive c1 and c2 based on diversity
            if e > 0:  # Diversity too low
                c1 = c1_max  # 2.0
                c2 = c2_min  # 1.0
            else:  # Diversity too high
                c1 = c1_min  # 1.0
                c2 = c2_max  # 2.0
            
            for i in range(num_particles):
                r1, r2 = np.random.random(particle_dim), np.random.random(particle_dim)
                velocities[i] = w * velocities[i] + \
                              c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                              c2 * r2 * (global_best_position - particles[i])
                
                # Transfer function with bounds checking
                try:
                    tf_values = transfer_functions[tf_name]["func"](velocities[i])
                except:
                    tf_values = np.zeros(particle_dim)
                
                # Flip-based strategy with validation
                particles[i] = np.where(np.random.random(particle_dim) < tf_values,
                                      1 - particles[i],  # Flip bit
                                      particles[i])
                
                # Evaluate with error handling
                try:
                    current_fitness[i] = evaluate_particle(particles[i], func_info)
                except:
                    current_fitness[i] = float('inf')
                
                # Update personal best if valid
                if current_fitness[i] < personal_best_scores[i] and current_fitness[i] != float('inf'):
                    personal_best_scores[i] = current_fitness[i]
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best if valid
                    if current_fitness[i] < global_best_score:
                        global_best_score = current_fitness[i]
                        global_best_position = particles[i].copy()
                        global_best_idx = i
            
            all_best_fitness[run, iter] = global_best_score
            all_diversity[run, iter] = D0
            all_c1_values[run, iter] = c1
            all_c2_values[run, iter] = c2
        
        all_final_fitness[run] = global_best_score
        execution_times[run] = time.time() - start_time
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    avg_time = np.mean(execution_times)
    avg_diversity = np.mean(all_diversity, axis=0)
    avg_c1 = np.mean(all_c1_values, axis=0)
    avg_c2 = np.mean(all_c2_values, axis=0)
    
    return avg_best_fitness, avg_final_fitness, std_final_fitness, avg_time, avg_diversity, avg_c1, avg_c2

def algo3_binary_pso(func_info, num_particles=30, max_iter=500, runs=20, 
                    tf_name="algo3", div_algo="algo1", div_params={'a':1.0, 'b':0.5}):
    """Algorithm 3 with linear parameter adaptation"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    w_initial, w_final = 1.0, 0.1
    
    all_best_fitness = np.zeros((runs, max_iter))
    all_final_fitness = np.zeros(runs)
    execution_times = np.zeros(runs)
    all_diversity = np.zeros((runs, max_iter))
    all_c1_values = np.zeros((runs, max_iter))
    all_c2_values = np.zeros((runs, max_iter))
    
    for run in range(runs):
        start_time = time.time()
        
        particles = initialize_particles(func_info, num_particles)
        velocities = np.random.uniform(-6, 6, (num_particles, particle_dim))
        
        current_fitness = np.array([evaluate_particle(p, func_info) for p in particles])
        personal_best_positions = particles.copy()
        personal_best_scores = current_fitness.copy()
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        for iter in range(max_iter):
            w = w_initial - (w_initial - w_final) * iter / max_iter  # Decreasing inertia
            
            # Linear adaptation of c1 and c2
            c1 = 2.0 - (1.0 * iter / max_iter)  # Decreases from 2 to 1
            c2 = 1.0 + (1.0 * iter / max_iter)  # Increases from 1 to 2
            
            for i in range(num_particles):
                r1, r2 = np.random.random(particle_dim), np.random.random(particle_dim)
                velocities[i] = w * velocities[i] + \
                              c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                              c2 * r2 * (global_best_position - particles[i])
                
                # Transfer function
                tf_values = transfer_functions[tf_name]["func"](velocities[i])
                
                # Flip-based strategy
                particles[i] = np.where(np.random.random(particle_dim) < tf_values,
                                      1 - particles[i],  # Flip bit
                                      particles[i])
                
                current_fitness[i] = evaluate_particle(particles[i], func_info)
                
                if current_fitness[i] < personal_best_scores[i]:
                    personal_best_scores[i] = current_fitness[i]
                    personal_best_positions[i] = particles[i].copy()
                    
                    if current_fitness[i] < global_best_score:
                        global_best_score = current_fitness[i]
                        global_best_position = particles[i].copy()
            
            all_best_fitness[run, iter] = global_best_score
            all_diversity[run, iter] = calculate_diversity(particles)
            all_c1_values[run, iter] = c1
            all_c2_values[run, iter] = c2
        
        all_final_fitness[run] = global_best_score
        execution_times[run] = time.time() - start_time
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    avg_time = np.mean(execution_times)
    avg_diversity = np.mean(all_diversity, axis=0)
    avg_c1 = np.mean(all_c1_values, axis=0)
    avg_c2 = np.mean(all_c2_values, axis=0)
    
    return avg_best_fitness, avg_final_fitness, std_final_fitness, avg_time, avg_diversity, avg_c1, avg_c2

def algo4_binary_pso(func_info, num_particles=30, max_iter=500, runs=20, 
                    tf_name="algo4", div_algo="algo1", div_params={'a':1.0, 'b':0.5}):
    """Algorithm 4 with sigmoid parameter adaptation"""
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    w_initial, w_final = 1.0, 0.1
    
    all_best_fitness = np.zeros((runs, max_iter))
    all_final_fitness = np.zeros(runs)
    execution_times = np.zeros(runs)
    all_diversity = np.zeros((runs, max_iter))
    all_c1_values = np.zeros((runs, max_iter))
    all_c2_values = np.zeros((runs, max_iter))
    
    for run in range(runs):
        start_time = time.time()
        
        particles = initialize_particles(func_info, num_particles)
        velocities = np.random.uniform(-6, 6, (num_particles, particle_dim))
        
        current_fitness = np.array([evaluate_particle(p, func_info) for p in particles])
        personal_best_positions = particles.copy()
        personal_best_scores = current_fitness.copy()
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        for iter in range(max_iter):
            w = w_initial - (w_initial - w_final) * iter / max_iter  # Decreasing inertia
            
            # Sigmoid adaptation of c1 and c2
            progress = iter / max_iter
            c1 = 2.0 - expit(10 * (progress - 0.5))  # Decreases from 2 to 1
            c2 = 1.0 + expit(10 * (progress - 0.5))  # Increases from 1 to 2
            
            for i in range(num_particles):
                r1, r2 = np.random.random(particle_dim), np.random.random(particle_dim)
                velocities[i] = w * velocities[i] + \
                              c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                              c2 * r2 * (global_best_position - particles[i])
                
                # Transfer function
                tf_values = transfer_functions[tf_name]["func"](velocities[i])
                
                # Flip-based strategy
                particles[i] = np.where(np.random.random(particle_dim) < tf_values,
                                      1 - particles[i],  # Flip bit
                                      particles[i])
                
                current_fitness[i] = evaluate_particle(particles[i], func_info)
                
                if current_fitness[i] < personal_best_scores[i]:
                    personal_best_scores[i] = current_fitness[i]
                    personal_best_positions[i] = particles[i].copy()
                    
                    if current_fitness[i] < global_best_score:
                        global_best_score = current_fitness[i]
                        global_best_position = particles[i].copy()
            
            all_best_fitness[run, iter] = global_best_score
            all_diversity[run, iter] = calculate_diversity(particles)
            all_c1_values[run, iter] = c1
            all_c2_values[run, iter] = c2
        
        all_final_fitness[run] = global_best_score
        execution_times[run] = time.time() - start_time
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    avg_time = np.mean(execution_times)
    avg_diversity = np.mean(all_diversity, axis=0)
    avg_c1 = np.mean(all_c1_values, axis=0)
    avg_c2 = np.mean(all_c2_values, axis=0)
    
    return avg_best_fitness, avg_final_fitness, std_final_fitness, avg_time, avg_diversity, avg_c1, avg_c2

def generate_comparison_plots(func_name, algo_results):
    """Generate comparison plots for all algorithms"""
    # Convergence plot
    plt.figure(figsize=(12, 6))
    for algo_name, results in algo_results.items():
        plt.plot(results["avg_fitness"], label=f"{algo_name} (Final: {results['avg_final']:.2f}±{results['std_final']:.2f})")
    plt.title(f"{func_name}\nConvergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save convergence plot
    base_filename = os.path.join(convergence_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_comparison.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_comparison.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Diversity plot with exploration/exploitation ratio
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    
    for algo_name, results in algo_results.items():
        div_max = np.max(results["avg_diversity"])
        if div_max > 0:
            exploration = (results["avg_diversity"] / div_max) * 100
            exploitation = (1 - results["avg_diversity"] / div_max) * 100
        else:
            exploration = np.zeros_like(results["avg_diversity"])
            exploitation = np.ones_like(results["avg_diversity"]) * 100
        
        ax1.plot(results["avg_diversity"], label=f"{algo_name} diversity")
        
        # Add exploration/exploitation ratio to legend
        avg_exploration = np.mean(exploration)
        avg_exploitation = np.mean(exploitation)
        plt.plot([], [], ' ', label=f"{algo_name} - Expl: {avg_exploration:.1f}%, Exploit: {avg_exploitation:.1f}%")
    
    plt.title(f"{func_name}\nDiversity Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Population Diversity")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save diversity plot
    base_filename = os.path.join(diversity_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_diversity_comparison.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_diversity_comparison.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Parameter adaptation plots for algorithms that have them
    for algo_name, results in algo_results.items():
        if "avg_c1" in results and "avg_c2" in results:
            plt.figure(figsize=(12, 6))
            plt.plot(results["avg_c1"], label="c1 (cognitive)")
            plt.plot(results["avg_c2"], label="c2 (social)")
            plt.title(f"{func_name}\n{algo_name} Parameter Adaptation")
            plt.xlabel("Iteration")
            plt.ylabel("Parameter Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            base_filename = os.path.join(diversity_dir, f"{func_name.replace(':', '_')}_{algo_name}")
            plt.savefig(f"{base_filename}_parameter_adaptation.svg", format="svg", bbox_inches='tight')
            plt.savefig(f"{base_filename}_parameter_adaptation.png", format="png", dpi=300, bbox_inches='tight')
            plt.close()

def save_comparison_results(func_name, algo_results):
    """Save comparison results to text file"""
    with open(os.path.join(results_dir, f"{func_name.replace(':', '_')}_comparison.txt"), "w") as f:
        f.write(f"Comparison Results for {func_name}\n")
        f.write("="*80 + "\n")
        f.write(f"{'Metric':<30}{'algo1':<25}{'algo2':<25}{'algo3':<25}{'algo4':<25}\n")
        f.write("-"*80 + "\n")
        
        # Write header with algorithm names
        algo_names = list(algo_results.keys())
        header = f"{'Metric':<30}"
        for name in algo_names:
            header += f"{name:<25}"
        f.write(header + "\n")
        f.write("-"*80 + "\n")
        
        # Write data
        f.write(f"{'Final Fitness (mean)':<30}")
        for name in algo_names:
            f.write(f"{algo_results[name]['avg_final']:<25.4f}")
        f.write("\n")
        
        f.write(f"{'Final Fitness (std)':<30}")
        for name in algo_names:
            f.write(f"{algo_results[name]['std_final']:<25.4f}")
        f.write("\n")
        
        f.write(f"{'Execution Time (s)':<30}")
        for name in algo_names:
            f.write(f"{algo_results[name]['avg_time']:<25.4f}")
        f.write("\n")
        
        f.write("\nConvergence Data:\n")
        f.write("Iteration," + ",".join(algo_names) + "\n")
        for i in range(len(next(iter(algo_results.values()))["avg_fitness"])):
            f.write(f"{i}")
            for name in algo_names:
                f.write(f",{algo_results[name]['avg_fitness'][i]:.4f}")
            f.write("\n")

def generate_heatmaps(global_results):
    """Generate heatmaps comparing all algorithms without function names"""
    num_functions = len(global_results)
    metrics = ["Final Fitness", "Execution Time", "Final Diversity"]
    
    # Get all unique algorithm combinations from the results
    all_algos = set()
    for func_results in global_results.values():
        all_algos.update(func_results.keys())
    all_algos = sorted(all_algos)  # Sort for consistent ordering
    
    for metric in metrics:
        plt.figure(figsize=(15, 10))
        
        # Prepare data matrix
        data = np.zeros((num_functions, len(all_algos)))
        for i, (func_name, results) in enumerate(global_results.items()):
            for j, algo in enumerate(all_algos):
                if algo in results:
                    if metric == "Final Fitness":
                        data[i, j] = results[algo]["avg_final"]
                    elif metric == "Execution Time":
                        data[i, j] = results[algo]["avg_time"]
                    elif metric == "Final Diversity":
                        data[i, j] = results[algo]["avg_diversity"][-1]
                else:
                    data[i, j] = np.nan  # Mark missing combinations
        
        # Normalize data (0=best, 1=worst for each function)
        if metric != "Final Diversity":
            min_vals = np.nanmin(data, axis=1, keepdims=True)
            max_vals = np.nanmax(data, axis=1, keepdims=True)
            norm_data = (data - min_vals) / (max_vals - min_vals + 1e-10)
        else:  # For diversity, higher might be better
            min_vals = np.nanmin(data, axis=1, keepdims=True)
            max_vals = np.nanmax(data, axis=1, keepdims=True)
            norm_data = (max_vals - data) / (max_vals - min_vals + 1e-10)
        
        # Create heatmap
        heatmap = plt.imshow(norm_data, cmap='RdYlGn_r', aspect='auto')
        
        # Add colorbar with meaningful labels
        cbar = plt.colorbar(heatmap, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Best', 'Medium', 'Worst'])
        cbar.set_label('Relative Performance')
        
        # Set ticks and labels
        plt.xticks(np.arange(len(all_algos)), all_algos, rotation=45)
        plt.yticks(np.arange(num_functions), [f"F{i+1}" for i in range(num_functions)])
        plt.xlabel('Algorithm Combinations')
        plt.ylabel('Test Functions')
        
        # Add performance ratio text (compared to best for each function)
        for i in range(num_functions):
            best_val = np.nanmin(data[i]) if metric != "Final Diversity" else np.nanmax(data[i])
            for j in range(len(all_algos)):
                if not np.isnan(data[i, j]):
                    ratio = data[i, j] / best_val if metric != "Final Diversity" else best_val / data[i, j]
                    
                    # Format text based on metric
                    if metric == "Final Fitness":
                        plt.text(j, i, f"{data[i,j]:.1f}\n({ratio:.1f}x)", 
                                ha="center", va="center", color="black", fontsize=8)
                    elif metric == "Execution Time":
                        plt.text(j, i, f"{data[i,j]:.2f}s\n({ratio:.1f}x)", 
                                ha="center", va="center", color="black", fontsize=8)
                    else:
                        plt.text(j, i, f"{data[i,j]:.2f}\n({ratio:.1f}x)", 
                                ha="center", va="center", color="black", fontsize=8)
        
        plt.title(f"Algorithm Comparison: {metric}\n(Lower values better for Fitness/Time)")
        plt.tight_layout()
        
        # Save heatmap
        heatmap_filename = os.path.join(heatmap_dir, f"{metric.replace(' ', '_')}_heatmap.png")
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {metric} heatmap to {heatmap_filename}")

def create_summary_table(global_results):
    """Create a summary table image with all results and statistical tests"""
    # Get all algorithms and functions
    all_algos = sorted({algo for func_results in global_results.values() for algo in func_results.keys()})
    all_funcs = sorted(global_results.keys())
    
    # Prepare data for the table
    table_data = []
    for func in all_funcs:
        row = [func]
        for algo in all_algos:
            if algo in global_results[func]:
                res = global_results[func][algo]
                row.append(f"{res['avg_final']:.2f} ± {res['std_final']:.2f}"
)
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Perform Wilcoxon rank-sum tests between all algorithm pairs
    p_values = {}
    for i, algo1 in enumerate(all_algos):
        for j, algo2 in enumerate(all_algos):
            if i >= j:
                continue  # Only compare each pair once
                
            # Collect all results for these two algorithms
            algo1_results = []
            algo2_results = []
            for func in all_funcs:
                if algo1 in global_results[func] and algo2 in global_results[func]:
                    algo1_results.append(global_results[func][algo1]['avg_final'])
                    algo2_results.append(global_results[func][algo2]['avg_final'])
            
            if len(algo1_results) > 0 and len(algo2_results) > 0:
                _, p_value = ranksums(algo1_results, algo2_results)
                p_values[(algo1, algo2)] = p_value
    
    # Create figure for the table
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create table with results
    table = ax.table(cellText=table_data, 
                    colLabels=["Function"] + all_algos,
                    loc='center', 
                    cellLoc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight best results in each row
    for i, row in enumerate(table_data):
        if i == 0:
            continue  # Skip header row
        
        best_val = None
        best_col = None
        for j, cell in enumerate(row[1:], start=1):
            if cell != "N/A":
                val = float(cell.split('±')[0].strip())
                if best_val is None or val < best_val:
                    best_val = val
                    best_col = j
        
        if best_col is not None:
            table[(i+1, best_col)].set_text_props(weight='bold')
    
    # Add statistical test results
    stat_text = "\nWilcoxon Rank-Sum Test p-values (5% significance level):\n"
    for (algo1, algo2), p_val in p_values.items():
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        stat_text += f"{algo1} vs {algo2}: p = {p_val:.4f} ({significance})\n"
    
    plt.figtext(0.5, 0.05, stat_text, ha="center", fontsize=10)
    
    plt.title("Algorithm Comparison Results (Best results in bold, N/A = Not Applicable)", y=1.05)
    
    # Save table as JPG
    table_filename = os.path.join(table_dir, "summary_table.jpg")
    plt.savefig(table_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary table to {table_filename}")

def run_all_combinations():
    results = {}
    
    # Define all combinations to test
    combinations = [
        # ("v4", "algo1", "algo2"),
        # ("v4", "algo2", "algo2"),
        # ("v4", "algo3", "algo2"),
        # ("v4", "algo4", "algo2"),
        # ("v5", "algo1", "algo2"),
        # ("v5", "algo2", "algo2"),
        # ("v5", "algo3", "algo2"),
        # ("v5", "algo4", "algo2"),
        ("v6", "algo1", "algo2"),
        ("v6", "algo2", "algo2"),
        ("v6", "algo3", "algo2"),
        ("v6", "algo4", "algo2"),
    ]
    
    for func_info in benchmark_functions:
        func_name = func_info["name"]
        results[func_name] = {}
        
        for tf_name, algo_name, div_algo in combinations:
            print(f"\nRunning {func_name} with TF={tf_name}, Algo={algo_name}, Div={div_algo}")
            
            if algo_name == "algo1":
                res = algo1_binary_pso(
                    func_info, 
                    tf_name=tf_name,
                    div_algo=div_algo,
                    div_params={"a": 1.0, "b": 0.5}
                )
                results[func_name][f"{tf_name}_{algo_name}"] = {
                    "avg_fitness": res[0],
                    "avg_final": res[1],
                    "std_final": res[2],
                    "avg_time": res[3],
                    "avg_diversity": res[4]
                }
            elif algo_name == "algo2":
                res = algo2_binary_pso(
                    func_info,
                    tf_name=tf_name,
                    div_algo=div_algo,
                    div_params={"a": 1.0, "b": 0.5}
                )
                results[func_name][f"{tf_name}_{algo_name}"] = {
                    "avg_fitness": res[0],
                    "avg_final": res[1],
                    "std_final": res[2],
                    "avg_time": res[3],
                    "avg_diversity": res[4],
                    "avg_c1": res[5],
                    "avg_c2": res[6]
                }
            elif algo_name == "algo3":
                res = algo3_binary_pso(
                    func_info,
                    tf_name=tf_name,
                    div_algo=div_algo,
                    div_params={"a": 1.0, "b": 0.5}
                )
                results[func_name][f"{tf_name}_{algo_name}"] = {
                    "avg_fitness": res[0],
                    "avg_final": res[1],
                    "std_final": res[2],
                    "avg_time": res[3],
                    "avg_diversity": res[4],
                    "avg_c1": res[5],
                    "avg_c2": res[6]
                }
            elif algo_name == "algo4":
                res = algo4_binary_pso(
                    func_info,
                    tf_name=tf_name,
                    div_algo=div_algo,
                    div_params={"a": 1.0, "b": 0.5}
                )
                results[func_name][f"{tf_name}_{algo_name}"] = {
                    "avg_fitness": res[0],
                    "avg_final": res[1],
                    "std_final": res[2],
                    "avg_time": res[3],
                    "avg_diversity": res[4],
                    "avg_c1": res[5],
                    "avg_c2": res[6]
                }
    
    return results

def run_comparison_experiments():
    global_results = run_all_combinations()
    
    # Generate plots and heatmaps for all combinations
    for func_name, algo_results in global_results.items():
        generate_comparison_plots(func_name, algo_results)
        save_comparison_results(func_name, algo_results)
    
    generate_heatmaps(global_results)
    create_summary_table(global_results)
    return global_results

if __name__ == "__main__":
    print("Binary PSO Algorithm Comparison")
    print("Comparing algo1 (fixed params) vs algo2 (adaptive params) vs algo3 (linear) vs algo4 (sigmoid)")
    results = run_comparison_experiments()
    print("\nComparison complete! Results saved in:", main_output_dir)