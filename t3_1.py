# binary-encoded BPSO for 25 benchmark functions from CEC 2005 
# compares v shape flip based (decreasing) with s shape set based (increasing) transfer functions
# includes four variants of V6 transfer function

# v6 compaired with variant 1 and 4 


import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
from scipy.special import erf
from datetime import datetime

# Create main output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_output_dir = f"BPSO_Binary_Encoding_Results_{timestamp}"
os.makedirs(main_output_dir, exist_ok=True)

# Create subdirectories
convergence_dir = os.path.join(main_output_dir, "1_Convergence_Plots")
results_dir = os.path.join(main_output_dir, "2_Results_Data")
performance_dir = os.path.join(main_output_dir, "3_Performance_Plots")
velocity_dir = os.path.join(main_output_dir, "4_Velocity_Analysis")
bitflip_dir = os.path.join(main_output_dir, "5_BitFlip_Analysis")

for directory in [convergence_dir, results_dir, performance_dir, velocity_dir, bitflip_dir]:
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

# Benchmark functions (CEC 2005) - Representative subset
import numpy as np

benchmark_functions = [
    {"name": "F1: Shifted sphere", 
     "func": lambda x: np.sum((x - 50)**2) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F2: Shifted Schwefel", 
     "func": lambda x: np.sum((x - 50)**2) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F3: Shifted rotated elliptic", 
     "func": lambda x: np.sum([1000**(i/(len(x)-1)) * (x[i]-50)**2 for i in range(len(x))]) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F4: Shifted Schwefel with noise", 
     "func": lambda x: np.sum((x - 50)**2) * (1 + 0.1 * np.random.normal(0, 1)) + 450,
     "range": [-100, 100], "dim": 5, "fmin": 450},
    
    {"name": "F5: Schwefel global opt bound", 
     "func": lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))) + 310,
     "range": [-100, 100], "dim": 5, "fmin": 310},
    
    {"name": "F6: Shifted Rosenbrock", 
     "func": lambda x: np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2) + 390,
     "range": [-100, 100], "dim": 5, "fmin": 390},
    
    {"name": "F7: Shifted rotated Griewank", 
     "func": lambda x: np.sum((x - 300)**2)/4000 - np.prod([np.cos((x[i]-300)/np.sqrt(i+1)) for i in range(len(x))]) + 1 + 180,
     "range": [0, 600], "dim": 5, "fmin": 180},
    
    {"name": "F8: Shifted rotated Ackley", 
     "func": lambda x: -20*np.exp(-0.2*np.sqrt(np.sum((x-32)**2)/len(x))) - np.exp(np.sum(np.cos(2*np.pi*(x-32)))/len(x)) + 20 + np.e + 140,
     "range": [-32, 32], "dim": 5, "fmin": 140},
    
    {"name": "F9: Shifted Rastrigin", 
     "func": lambda x: 10*len(x) + np.sum((x-2.5)**2 - 10*np.cos(2*np.pi*(x-2.5))) + 330,
     "range": [-5, 5], "dim": 5, "fmin": 330},
    
    {"name": "F10: Shifted rotated Rastrigin", 
     "func": lambda x: 10*len(x) + np.sum((x-2.5)**2 - 10*np.cos(2*np.pi*(x-2.5))) + 330,
     "range": [-5, 5], "dim": 5, "fmin": 330},
    
    {"name": "F11: Shifted rotated Weierstrass", 
     "func": lambda x: sum([sum([0.5**k * np.cos(2*np.pi*3**k*(x[i]-0.25+0.5)) for k in range(21)]) for i in range(len(x))]) - len(x)*sum([0.5**k * np.cos(2*np.pi*3**k*0.5) for k in range(21)]) + 90,
     "range": [-0.5, 0.5], "dim": 5, "fmin": 90},
    
    {"name": "F12: Schwefel", 
     "func": lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))) + 460,
     "range": [-100, 100], "dim": 5, "fmin": 460},
    
    {"name": "F13: Expanded Griewank+Rosenbrock", 
     "func": lambda x: (1 + np.sum(x**2)/4000 - np.prod([np.cos(x[i]/np.sqrt(i+1)) for i in range(len(x))]) + np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2))/2 + 130,
     "range": [-3, 1], "dim": 5, "fmin": 130},
    
    {"name": "F14: Shifted rotated expanded Scaffer", 
     "func": lambda x: sum([0.5 + (np.sin(np.sqrt((x[i]-50)**2 + (x[(i+1)%len(x)]-50)**2))**2 - 0.5)/(1 + 0.001*((x[i]-50)**2 + (x[(i+1)%len(x)]-50)**2))**2 for i in range(len(x))]) + 300,
     "range": [-100, 100], "dim": 5, "fmin": 300},
    
    {"name": "F15: Hybrid composition", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 120,
     "range": [-5, 5], "dim": 10, "fmin": 120},
    
    {"name": "F16: Rotated hybrid composition", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 120,
     "range": [-5, 5], "dim": 10, "fmin": 120},
    
    {"name": "F17: Rotated hybrid composition with noise", 
     "func": lambda x: np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) * (1 + 0.1*np.random.normal(0, 1)) + 120,
     "range": [-5, 5], "dim": 10, "fmin": 120},
    
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
    
    {"name": "F22: Rotated hybrid composition with high condition number", 
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

def v6_transfer_function(v, variant_type, iteration=None, max_iter=None, velocity_history=None):
    """
    V6 transfer function variants
    Args:
        v: velocity value or array
        variant_type: 1-4 for different variants
        iteration: current iteration (for variants 2 and 3)
        max_iter: maximum iterations (for variants 2 and 3)
        velocity_history: array of velocity magnitudes (for variant 4)
    """
    if variant_type == 1:
        # Type 1 (Baseline): Constant n=1
        return 1 - np.exp((-1/2)*np.abs(v))
    elif variant_type == 2:
        # Type 2: Linearly decreasing n from 5 to 1
        n = 5 - 6 * (iteration / max_iter)
        return 1 - np.exp((-1/2)*np.abs(v)**n)
    elif variant_type == 3:
        # Type 3: Piecewise n=5 for first half, n=1 for second half
        n = 5 if iteration < max_iter/2 else 1
        return 1 - np.exp((-1/2)*np.abs(v)**n)
    elif variant_type == 4:
        # Type 4: Velocity-based n (5 if |v|>1, else 1)
        n = np.where(np.abs(v) > 2, 5, 1)
        return 1 - np.exp((-1/2)*np.abs(v)**n)
    else:
        raise ValueError("Invalid variant type for V6 transfer function")

# Transfer functions including V6 variants
transfer_functions = [
    # {"name": "V1", "func": lambda v: np.abs(erf(np.sqrt(np.pi)*v/2)), "type": "v", "inertia": "decreasing"},
    # {"name": "V2", "func": lambda v: np.ab(s(np.tanh(v))), "type": "v", "inertia": "decreasing"},
    # {"name": "V3", "func": lambda v: np.abs((v/np.sqrt(1+v**2))), "type": "v", "inertia": "decreasing"},
    # {"name": "V4", "func": lambda v: np.abs(((2/np.pi)*np.arctan(np.pi/2*v))), "type": "v", "inertia": "decreasing"},
    # {"name": "V5", "func": lambda v: (2/np.pi)*((np.arctan(v**2))), "type": "v", "inertia": "decreasing"},

    {"name": "V6_Type1", "func": lambda v, iter=None, max_iter=None: v6_transfer_function(v, 1, iter, max_iter), 
     "type": "v", "inertia": "decreasing", "variant": 1},
    # {"name": "V6_Type2", "func": lambda v, iter, max_iter: v6_transfer_function(v, 2, iter, max_iter), 
    #  "type": "v", "inertia": "decreasing", "variant": 2},
    # {"name": "V6_Type3", "func": lambda v, iter, max_iter: v6_transfer_function(v, 3, iter, max_iter), 
    #  "type": "v", "inertia": "decreasing", "variant": 3},
    {"name": "V6_Type4", "func": lambda v, iter=None, max_iter=None: v6_transfer_function(v, 4, iter, max_iter), 
     "type": "v", "inertia": "decreasing", "variant": 4},
]

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
    
    # Convert to NumPy array before evaluation
    return func_info["func"](np.array(real_position))

def binary_pso(func_info, transfer_func, num_particles=30, max_iter=500, runs=30):
    dim = func_info["dim"]
    search_range = func_info["range"]
    particle_dim = dim * BITS_PER_VAR
    c1, c2 = 1.49, 1.49
    w_initial, w_final = 1.1, 0.1
    
    all_best_fitness = np.zeros((runs, max_iter))
    all_final_fitness = np.zeros(runs)
    execution_times = np.zeros(runs)
    
    # For velocity and bit-flip analysis
    velocity_history = np.zeros((runs, max_iter))
    bitflip_history = np.zeros((runs, max_iter))
    
    for run in range(runs):
        start_time = time.time()
        
        # Initialize particles and velocities
        particles = initialize_particles(func_info, num_particles)
        velocities = np.random.uniform(-6, 6, (num_particles, particle_dim))
        
        # Evaluate initial positions
        current_fitness = np.array([evaluate_particle(p, func_info) for p in particles])
        personal_best_positions = particles.copy()
        personal_best_scores = current_fitness.copy()
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        for iter in range(max_iter):
            if transfer_func["inertia"] == "increasing":
                w = w_final + (w_initial - w_final) * iter / max_iter
            else:
                w = w_initial - (w_initial - w_final) * iter / max_iter
            
            # Track average velocity magnitude
            avg_velocity = np.mean(np.abs(velocities))
            velocity_history[run, iter] = avg_velocity
            
            bit_flips = 0
            
            for i in range(num_particles):
                # Update velocity
                r1, r2 = np.random.random(particle_dim), np.random.random(particle_dim)
                velocities[i] = w * velocities[i] + \
                              c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                              c2 * r2 * (global_best_position - particles[i])
                
                # Apply transfer function
                if "variant" in transfer_func:
                    if transfer_func["variant"] in [2, 3]:
                        # Variants that need iteration info
                        tf_values = transfer_func["func"](velocities[i], iter, max_iter)
                    else:
                        # Variants that don't need iteration info
                        tf_values = transfer_func["func"](velocities[i])
                else:
                    tf_values = transfer_func["func"](velocities[i])
                
                # Update position
                if transfer_func["type"] == "s":
                    # S-shaped transfer function
                    new_particles = np.where(np.random.random(particle_dim) < tf_values, 1, 0)
                    bit_flips += np.sum(new_particles != particles[i])
                    particles[i] = new_particles
                else:
                    # V-shaped transfer function
                    flip_mask = np.random.random(particle_dim) < tf_values
                    bit_flips += np.sum(flip_mask)
                    particles[i] = np.where(flip_mask, 1 - particles[i], particles[i])
                
                # Evaluate new position
                current_fitness[i] = evaluate_particle(particles[i], func_info)
                
                # Update personal best
                if current_fitness[i] < personal_best_scores[i]:
                    personal_best_scores[i] = current_fitness[i]
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if current_fitness[i] < global_best_score:
                        global_best_score = current_fitness[i]
                        global_best_position = particles[i].copy()
            
            all_best_fitness[run, iter] = global_best_score
            bitflip_history[run, iter] = bit_flips / (num_particles * particle_dim)  # Normalized bit-flip rate
        
        all_final_fitness[run] = global_best_score
        execution_times[run] = time.time() - start_time
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    avg_time = np.mean(execution_times)
    
    # Calculate average velocity and bit-flip history across runs
    avg_velocity_history = np.mean(velocity_history, axis=0)
    avg_bitflip_history = np.mean(bitflip_history, axis=0)
    
    return avg_best_fitness, avg_final_fitness, std_final_fitness, avg_time, avg_velocity_history, avg_bitflip_history

def generate_convergence_plot(func_name, tf_results):
    plt.figure(figsize=(12, 8))
    
    # Sort transfer functions by final performance
    sorted_tfs = sorted(tf_results.items(), key=lambda x: x[1]["avg_final"])
    
    # Plot all convergence curves
    for tf_name, data in sorted_tfs:
        plt.plot(data["avg_fitness"], label=f"{tf_name} (Final: {data['avg_final']:.2f}±{data['std_final']:.2f})")
    
    # Highlight best and worst
    plt.plot(sorted_tfs[0][1]["avg_fitness"], linewidth=3, color='green', 
            label=f"BEST: {sorted_tfs[0][0]} ({sorted_tfs[0][1]['avg_final']:.2f}±{sorted_tfs[0][1]['std_final']:.2f})")
    plt.plot(sorted_tfs[-1][1]["avg_fitness"], linewidth=3, color='red', 
            label=f"WORST: {sorted_tfs[-1][0]} ({sorted_tfs[-1][1]['avg_final']:.2f}±{sorted_tfs[-1][1]['std_final']:.2f})")
    
    plt.title(f"{func_name}\nConvergence Plot of All Transfer Functions", pad=20)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save plots
    base_filename = os.path.join(convergence_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_convergence.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_convergence.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_plot(func_name, tf_results):
    plt.figure(figsize=(14, 8))
    
    # Sort transfer functions by performance
    sorted_tfs = sorted(tf_results.items(), key=lambda x: x[1]["avg_final"])
    tf_names = [x[0] for x in sorted_tfs]
    final_values = [x[1]["avg_final"] for x in sorted_tfs]
    std_values = [x[1]["std_final"] for x in sorted_tfs]
    times = [x[1]["avg_time"] for x in sorted_tfs]
    
    # Create bar plot with error bars
    bars = plt.bar(range(len(tf_names)), final_values, yerr=std_values, 
                  capsize=5, alpha=0.7)
    
    # Highlight best and worst
    bars[0].set_color('green')
    bars[-1].set_color('red')
    
    # Add time information as text above bars
    for i, (value, time_val) in enumerate(zip(final_values, times)):
        plt.text(i, value + std_values[i] + 0.05*max(final_values), 
                f"{time_val:.2f}s", ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(len(tf_names)), tf_names, rotation=45)
    plt.xlabel("Transfer Function")
    plt.ylabel("Final Fitness Value")
    plt.title(f"{func_name}\nTransfer Function Performance Comparison\n"
             f"Best: {tf_names[0]} ({final_values[0]:.2f}±{std_values[0]:.2f}), "
             f"Worst: {tf_names[-1]} ({final_values[-1]:.2f}±{std_values[-1]:.2f})", pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save plots
    base_filename = os.path.join(performance_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_performance.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_performance.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_velocity_plot(func_name, tf_results):
    """Generate plot of velocity evolution across iterations"""
    plt.figure(figsize=(12, 8))
    
    for tf_name, data in tf_results.items():
        if "avg_velocity" in data:
            plt.plot(data["avg_velocity"], label=tf_name)
    
    plt.title(f"{func_name}\nVelocity Evolution Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Velocity Magnitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save plots
    base_filename = os.path.join(velocity_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_velocity.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_velocity.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_bitflip_plot(func_name, tf_results):
    """Generate plot of bit-flip probability across iterations"""
    plt.figure(figsize=(12, 8))
    
    for tf_name, data in tf_results.items():
        if "avg_bitflip" in data:
            plt.plot(data["avg_bitflip"], label=tf_name)
    
    plt.title(f"{func_name}\nBit-Flip Probability Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Bit-Flip Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save plots
    base_filename = os.path.join(bitflip_dir, func_name.replace(":", "_"))
    plt.savefig(f"{base_filename}_bitflip.svg", format="svg", bbox_inches='tight')
    plt.savefig(f"{base_filename}_bitflip.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

def save_text_results(func_name, tf_results):
    # Sort by performance
    sorted_tfs = sorted(tf_results.items(), key=lambda x: x[1]["avg_final"])
    
    with open(os.path.join(results_dir, f"{func_name.replace(':', '_')}_results.txt"), "w") as f:
        f.write(f"Results for {func_name}\n")
        f.write("="*80 + "\n")
        f.write(f"{'TF':<20}{'Final Fitness':<20}{'Std Dev':<15}{'Time (s)':<15}{'Performance'}\n")
        f.write("-"*80 + "\n")
        
        for i, (tf_name, data) in enumerate(sorted_tfs):
            performance = ""
            if i == 0:
                performance = "BEST"
            elif i == len(sorted_tfs) - 1:
                performance = "WORST"
            
            f.write(f"{tf_name:<20}{data['avg_final']:<20.4f}{data['std_final']:<15.4f}"
                    f"{data['avg_time']:<15.4f}{performance}\n")
        
        f.write("\n\nConvergence Data:\n")
        f.write("Iteration," + ",".join([tf[0] for tf in sorted_tfs]) + "\n")
        
        # Get number of iterations from one of the result entries
        num_iters = len(next(iter(tf_results.values()))["avg_fitness"])
        
        for i in range(num_iters):
            values = [f"{tf[1]['avg_fitness'][i]:.4f}" for tf in sorted_tfs]
            f.write(f"{i}," + ",".join(values) + "\n")
        
        # Save velocity and bit-flip data if available
        for tf_name, data in sorted_tfs:
            if "avg_velocity" in data:
                f.write(f"\n\nVelocity Data for {tf_name}:\n")
                f.write("Iteration,AvgVelocity\n")
                for i, vel in enumerate(data["avg_velocity"]):
                    f.write(f"{i},{vel:.6f}\n")
            
            if "avg_bitflip" in data:
                f.write(f"\n\nBit-Flip Data for {tf_name}:\n")
                f.write("Iteration,AvgBitFlipProbability\n")
                for i, prob in enumerate(data["avg_bitflip"]):
                    f.write(f"{i},{prob:.6f}\n")

def generate_performance_heatmap(global_results):
    # Prepare data for heatmap
    func_names = [f"F{i+1}" for i in range(len(benchmark_functions))]
    tf_names = [tf["name"] for tf in transfer_functions]
    
    # Create a matrix of performance values
    performance_matrix = np.zeros((len(func_names), len(tf_names)))
    
    for i, func_name in enumerate(global_results.keys()):
        for j, tf_name in enumerate(tf_names):
            performance_matrix[i, j] = global_results[func_name][tf_name]["avg_final"]
    
    # Normalize the performance matrix (0 = best, 1 = worst for each function)
    normalized_matrix = np.zeros_like(performance_matrix)
    for i in range(len(func_names)):
        min_val = np.min(performance_matrix[i])
        max_val = np.max(performance_matrix[i])
        if max_val != min_val:
            normalized_matrix[i] = (performance_matrix[i] - min_val) / (max_val - min_val)
    
    # Create the heatmap
    plt.figure(figsize=(16, 12))
    plt.imshow(normalized_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Normalized Performance (0=best, 1=worst for each function)')
    
    # Add labels
    plt.xticks(np.arange(len(tf_names)), tf_names, rotation=45)
    plt.yticks(np.arange(len(func_names)), func_names)
    plt.xlabel('Transfer Functions')
    plt.ylabel('Benchmark Functions')
    plt.title('Performance Heatmap of All Transfer Functions Across All Benchmark Functions')
    
    # Add text annotations
    for i in range(len(func_names)):
        for j in range(len(tf_names)):
            plt.text(j, i, f"{performance_matrix[i,j]:.1f}", 
                    ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(main_output_dir, "6_Performance_Heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {heatmap_path}")

def run_experiments():
    global_results = {}
    
    with open(os.path.join(results_dir, "00_Summary_Results.txt"), "w") as summary_file:
        summary_file.write("Binary-Encoded BPSO EXPERIMENT SUMMARY\n")
        summary_file.write(f"Bits per variable: {BITS_PER_VAR} (1 sign + {MAGNITUDE_BITS} magnitude)\n")
        
        for func_info in benchmark_functions:
            func_name = func_info["name"]
            print(f"\nRunning {func_name} (Binary Dim: {func_info['dim']*BITS_PER_VAR})...")
            
            tf_results = {}
            for tf in transfer_functions:
                print(f"  {tf['name']}...", end=" ", flush=True)
                avg_fitness, avg_final, std_final, avg_time, avg_velocity, avg_bitflip = binary_pso(func_info, tf)
                tf_results[tf["name"]] = {
                    "avg_fitness": avg_fitness,
                    "avg_final": avg_final,
                    "std_final": std_final,
                    "avg_time": avg_time,
                    "avg_velocity": avg_velocity,
                    "avg_bitflip": avg_bitflip
                }
                print(f"Fitness: {avg_final:.2f}±{std_final:.2f}")
            
            global_results[func_info["name"]] = tf_results
            
            # Generate plots and save results
            generate_convergence_plot(func_name, tf_results)
            generate_performance_plot(func_name, tf_results)
            generate_velocity_plot(func_name, tf_results)
            generate_bitflip_plot(func_name, tf_results)
            save_text_results(func_name, tf_results)
            
            # Write to summary file
            best_tf = min(tf_results.items(), key=lambda x: x[1]["avg_final"])
            worst_tf = max(tf_results.items(), key=lambda x: x[1]["avg_final"])
            
            summary_file.write(f"{func_name}\n")
            summary_file.write(f"  Best: {best_tf[0]} ({best_tf[1]['avg_final']:.2f}±{best_tf[1]['std_final']:.2f})\n")
            summary_file.write(f"  Worst: {worst_tf[0]} ({worst_tf[1]['avg_final']:.2f}±{worst_tf[1]['std_final']:.2f})\n")
            summary_file.write("-"*60 + "\n")
    
    # Generate performance heatmap after all experiments
    generate_performance_heatmap(global_results)
    
    return global_results

if __name__ == "__main__":
    print("Binary-Encoded BPSO for CEC 2005 Benchmark Functions")
    print(f"Each real variable encoded to {BITS_PER_VAR} bits (1 sign + {MAGNITUDE_BITS} magnitude)")
    results = run_experiments()