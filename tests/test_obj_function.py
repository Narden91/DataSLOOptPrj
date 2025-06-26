import numpy as np
import sys
import matplotlib.pyplot as plt
from feeder_mapping.generator import IdealGenerator
from feeder_mapping.objective import objective_function_squared_sum
from feeder_mapping.utils import generate_all_connection_vectors

sys.dont_write_bytecode = True

def plot_results(results):
    """Generates and displays a plot of the sorted objective function scores."""
    
    # Sort results by score in ascending order
    results.sort(key=lambda x: x['score'])

    scores = [r['score'] for r in results]
    # Convert solution vectors to strings for labeling
    solutions_str = [str(r['solution']) for r in results]

    print("\n--- Plotting Results ---")
    print(f"Lowest score: {scores[0]:.4f} for solution {solutions_str[0]}")
    print(f"Highest score: {scores[-1]:.4f} for solution {solutions_str[-1]}")

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


def run_bruteforce_test(objective_function, nc=4, nf=4, wires_per_feeder=3, nt=10, seed=42):
    """
    Performs a brute-force search on a small-scale problem, validates
    the objective function, and plots the results.
    """
    # 1. Define parameters for a small, solvable problem
    # nc -> Number of consumers (apartments)
    # nf -> Number of feeders (cables)
    # wires_per_feeder -> Wires per cable (e.g., 3-phase)
    # nt -> Number of time steps

    nw = nf * wires_per_feeder # Total wires in the system

    print("--- Starting Brute-Force Objective Function Test ---")
    print(f"Parameters: nc={nc}, nf={nf}, nt={nt}, nw={nw}, wires_per_feeder={wires_per_feeder}")
    print(f"Total combinations to check: {nw**nc} ({nw}^{nc})")
    print("-" * 50)

    # 2. Generate ideal data based on a known ground truth
    generator = IdealGenerator(nc=nc, nf=nw, nt=nt, data_path="../data/clean_data.csv")
    meter_supply, line_supply, _, ground_truth_feeder_topo = generator.generate(seed=seed)

    print(f"📊 Generated Meter Data Shape: {meter_supply.shape}")
    print(f"🔌 Generated Line Data Shape: {line_supply.shape}")
    print(f"🧩 Ground Truth Feeder Topology: {ground_truth_feeder_topo}\n")

    # 3. Iterate through every possible solution vector and store results
    print("Evaluating all possible wire assignments...")
    all_possible_solutions = generate_all_connection_vectors(
        n_apartments=nc, 
        n_phases=nw  # 'n_phases' is the number of wire choices
    )

    results = []
    n_of_solutions = nw**nc
    for i, proposed_solution in enumerate(all_possible_solutions):
        if (i+1)%20000==0:
            print(f"Progress: {i+1} / {n_of_solutions}")
        score = objective_function(
            solution=proposed_solution,
            nc=nc, nf=nw, meters=meter_supply, lines=line_supply,
            wires_per_feeder=wires_per_feeder
        )
        results.append({'solution': proposed_solution, 'score': score})
    
    print("...Evaluation complete.\n")

    # 4. Plot the results
    plot_results(results)


if __name__ == "__main__":
    run_bruteforce_test(objective_function=objective_function_squared_sum, nc = 5, nf = 3, wires_per_feeder = 3, nt = 10, seed = 42)