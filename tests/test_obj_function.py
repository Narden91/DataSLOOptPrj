import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
    
sys.dont_write_bytecode = True

from feeder_mapping import IdealGenerator, objective_function_squared_sum, generate_all_connection_vectors


def plot_results(results):
    """Generates and displays a plot of the sorted objective function scores."""
    
    # Sort results by score in ascending order
    results.sort(key=lambda x: x['score'])

    scores = [r['score'] for r in results]
    # Convert solution vectors to strings for labeling
    solutions_str = [str(r['solution']) for r in results]

    print("\n" + "="*60)
    print("📊 PLOTTING RESULTS")
    print("="*60)
    print(f"🏆 Best score:  {scores[0]:.6f} → Solution: {solutions_str[0]}")
    print(f"💥 Worst score: {scores[-1]:.6f} → Solution: {solutions_str[-1]}")
    print(f"📈 Score range: {scores[-1] - scores[0]:.6f}")
    print("="*60)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    num_results = len(results)
    x_axis = np.arange(num_results)

    # Use a bar chart for few results, a line plot for many
    if num_results <= 100:
        colors = ['#1f77b4'] * num_results  # Default blue
        # Highlight all bars that have the minimum score
        min_score = scores[0]
        for i, score in enumerate(scores):
            if np.isclose(score, min_score):
                colors[i] = '#2ca02c' # Green for best
        
        ax.bar(x_axis, scores, color=colors)
        # Add a custom legend entry for the best score
        ax.bar(0, 0, color='#2ca02c', label=f'Best Solution(s) (Score ≈ {min_score:.2f})')
        ax.set_xticks([]) # Hide x-ticks as they would be too cluttered
    else:
        ax.plot(x_axis, scores, marker='.', linestyle='-', markersize=4, label='Score')
        # Highlight the minimum point
        ax.plot(0, scores[0], 'go', markersize=8, label=f'Best Solution (Score ≈ {scores[0]:.2f})')

    # Annotate the best and worst solutions
    best_text = f"Best: {solutions_str[0]}"
    ax.annotate(best_text, xy=(0, scores[0]), xytext=(15, 20),
                textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="g", lw=1, alpha=0.9))

    worst_text = f"Worst: {solutions_str[-1]}"
    ax.annotate(worst_text, xy=(num_results - 1, scores[-1]), xytext=(-80, -30),
                textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="r", lw=1, alpha=0.9))

    ax.set_title(f'Objective Function Scores for All {num_results} Possible Solutions (Sorted)', fontsize=16)
    ax.set_xlabel('Each Solution Vector (Sorted by Score)', fontsize=12)
    ax.set_ylabel('Objective Score (Sum of Squared Errors)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Save the plot to a file in plots directory with a timestamp
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(plots_dir, f"bruteforce_results_{timestamp}.png")
    fig.savefig(plot_filename, dpi=300)
    print(f"💾 Plot saved to: {plot_filename}\n")


def run_bruteforce_test(objective_function, nc=4, nf=4, wires_per_feeder=3, nt=10, seed=42):
    """
    Performs a brute-force search on a small-scale problem, validates
    the objective function, and plots the results.
    
    objective_function: The function to evaluate solutions.
    nc: Number of consumers (apartments).
    nf: Number of feeders (cables).
    wires_per_feeder: Number of wires per feeder (e.g., 3 for 3-phase).
    nt: Number of time steps.
    seed: Random seed for reproducibility.
    
    """
    # 1. Calculate the total number of wires in the system
    nw = nf * wires_per_feeder 

    print("\n" + "="*70)
    print("🚀 BRUTE-FORCE OBJECTIVE FUNCTION TEST")
    print("="*70)
    print(f"🏠 Consumers (apartments): {nc}")
    print(f"🔌 Feeders (cables):       {nf}")
    print(f"⚡ Wires per feeder:       {wires_per_feeder}")
    print(f"🕐 Time steps:             {nt}")
    print(f"🔢 Total wires:            {nw}")
    print(f"🎯 Random seed:            {seed}")
    print(f"🧮 Total combinations:     {nw**nc:,} ({nw}^{nc})")
    print("="*70)

    # 2. Generate ideal data based on a known ground truth
    print("\n🔧 Generating synthetic data...")
    generator = IdealGenerator(nc=nc, nf=nw, nt=nt, data_path="data/clean_data.csv")
    meter_supply, line_supply, _, ground_truth_feeder_topo = generator.generate(seed=seed)

    print(f"✅ Data generation complete!")
    print(f"   📊 Meter data shape:      {meter_supply.shape}")
    print(f"   🔌 Line data shape:       {line_supply.shape}")
    print(f"   🧩 Ground truth topology: {ground_truth_feeder_topo}")

    # 3. Iterate through every possible solution vector and store results
    print(f"\n🔍 Evaluating all {nw**nc:,} possible wire assignments...")
    all_possible_solutions = generate_all_connection_vectors(
        n_apartments=nc, 
        n_phases=nw  # 'n_phases' is the number of wire choices
    )

    results = []
    n_of_solutions = nw**nc
    
    # Use tqdm for progress bar
    with tqdm(total=n_of_solutions, desc="🔄 Processing", unit="solutions", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for i, proposed_solution in enumerate(all_possible_solutions):
            score = objective_function(
                solution=proposed_solution,
                nc=nc, nf=nw, meters=meter_supply, lines=line_supply,
                wires_per_feeder=wires_per_feeder
            )
            results.append({'solution': proposed_solution, 'score': score})
            pbar.update(1)
            
            # Update description with current best score every 1000 iterations
            if (i + 1) % 1000 == 0 and results:
                current_best = min(results, key=lambda x: x['score'])['score']
                pbar.set_description(f"🔄 Processing (best: {current_best:.4f})")
    
    print("✅ Evaluation complete!")

    # 4. Plot the results
    plot_results(results)


if __name__ == "__main__":
    nc= 5  # Number of consumers (apartments)
    nf = 3  # Number of feeders (cables)
    wires_per_feeder = 3  # Number of wires per feeder (e.g., 3 for 3-phase)
    nt = 10  # Number of time steps (not used in this test)
    seed = 42  # Random seed for reproducibility
    
    run_bruteforce_test(
        objective_function=objective_function_squared_sum,
        nc=nc, nf=nf, wires_per_feeder=wires_per_feeder, nt=nt, seed=seed
    )