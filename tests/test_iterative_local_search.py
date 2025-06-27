import sys
sys.dont_write_bytecode = True

from feeder_mapping.generator import IdealGenerator
from feeder_mapping.objective import objective_function_squared_sum
from heuristics.neighborhoods import insert_neighborhood
from heuristics.local_search import iterative_local_search, simulated_annealing_temperature, adaptive_perturbation
from heuristics.utils import random_perturbation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_iterative_local_search(objective_function, generate_neighbors, perturbation, data_path,
                               nc, nf, nt, seed, wires_per_feeder, max_iter, max_ils_iter):
    np.random.seed(seed)

    nw = nf * wires_per_feeder
    print(f"🔧 Initializing IdealGenerator with {nc} consumers, {nf} feeders, and {nt} time steps...")

    # Fix: Pass nf (not nw) to generator since it expects number of feeders
    generator = IdealGenerator(nc=nc, nf=nf, nt=nt, data_path=data_path)
    meter_supply, line_supply, _, topology = generator.generate(seed=seed)

    print("📊 Meter Supply Shape:", meter_supply.shape)
    print("🔌 Line Supply Shape:", line_supply.shape)
    print("🧩 Ground Truth Feeder Topology:", topology)
    print(f"🧮 System has {nw} wires ({wires_per_feeder} per feeder)\n")

    random_solution = np.random.randint(0, nw, size=nc)
    initial_score = objective_function(
        solution=random_solution,
        nc=nc,
        nf=nf,  # Fix: Pass nf (number of feeders, not wires)
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder
    )
    print(f"🎯 Initial score: {initial_score:.4f}")

    # Use improved iterative local search with multiple strategies
    print("\n🚀 Starting improved iterative local search...")
    
    best_solution, best_objective = iterative_local_search(
        objective_function=objective_function,
        generate_neighbors=generate_neighbors,
        perturbation=lambda sol: adaptive_perturbation(sol, base_changes=3, nw=nw),
        initial_solution=random_solution,
        nc=nc,
        nf=nf,  # Fix: Pass nf (number of feeders, not wires)
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder,
        max_iterations=max_iter,
        max_ils_iterations=max_ils_iter,
        verbose=True,
        acceptance_criterion="better",  # Can be "better", "simulated_annealing", "threshold"
        early_stopping_patience=None,  # Disable early stopping
        improvement_threshold=1e-8
    )

    print("\n" + "="*60)
    print("🧩 Ground Truth Feeder Topology:", topology)
    print("🏁 Final solution (first 10):", best_solution[:10])
    print(f"🏆 Final objective score: {best_objective:.6f}")
    
    # Convert wire assignments to feeder assignments for comparison
    final_feeder_assignments = best_solution // wires_per_feeder
    print("🔌 Final feeder assignments (first 10):", final_feeder_assignments[:10])
    
    # Calculate accuracy if possible
    if len(topology) == len(final_feeder_assignments):
        accuracy = np.mean(topology == final_feeder_assignments)
        print(f"🎯 Assignment accuracy: {accuracy:.2%}")
    
    return best_solution, best_objective


def run_single_algorithm(strategy, data_path, nc, nf, nt, seed, wires_per_feeder, max_iter, max_ils_iter):
    """Run a single algorithm with given parameters."""
    np.random.seed(seed)
    nw = nf * wires_per_feeder
    
    generator = IdealGenerator(nc=nc, nf=nf, nt=nt, data_path=data_path)
    meter_supply, line_supply, _, topology = generator.generate(seed=seed)
    
    random_solution = np.random.randint(0, nw, size=nc)
    
    best_solution, best_objective = iterative_local_search(
        objective_function=objective_function_squared_sum,
        generate_neighbors=insert_neighborhood,
        perturbation=strategy["perturbation"],
        initial_solution=random_solution,
        nc=nc,
        nf=nf,
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder,
        max_iterations=max_iter,
        max_ils_iterations=max_ils_iter,
        verbose=False,
        acceptance_criterion=strategy["acceptance"],
        temperature_schedule=strategy["temperature"],
        early_stopping_patience=None,  # Disable early stopping
        improvement_threshold=1e-8
    )
    
    return best_objective, best_solution


def run_multiple_runs_comparison(data_path, nc, nf, nt, base_seed, wires_per_feeder, 
                                max_iter, max_ils_iter, num_runs=20):
    """Run multiple independent runs for each algorithm and collect results for boxplot."""
    print("\n" + "="*80)
    print("🎯 RUNNING MULTIPLE INDEPENDENT RUNS FOR STATISTICAL COMPARISON")
    print("="*80)
    
    strategies = [
        {
            "name": "Standard ILS",
            "acceptance": "better",
            "perturbation": lambda sol: random_perturbation(sol, num_changes=3, nw=nf * wires_per_feeder),
            "temperature": None
        },
        {
            "name": "Simulated Annealing ILS",
            "acceptance": "simulated_annealing", 
            "perturbation": lambda sol: random_perturbation(sol, num_changes=2, nw=nf * wires_per_feeder),
            "temperature": simulated_annealing_temperature
        },
        {
            "name": "High Perturbation ILS",
            "acceptance": "better",
            "perturbation": lambda sol: random_perturbation(sol, num_changes=5, nw=nf * wires_per_feeder),
            "temperature": None
        },
        {
            "name": "Threshold ILS",
            "acceptance": "threshold",
            "perturbation": lambda sol: random_perturbation(sol, num_changes=3, nw=nf * wires_per_feeder),
            "temperature": None
        }
    ]
    
    # Collect results for all algorithms
    all_results = []
    
    for strategy in strategies:
        print(f"\n🧪 Running {num_runs} independent runs for {strategy['name']}...")
        strategy_results = []
        
        # Progress bar for runs
        from tqdm import tqdm
        for run in tqdm(range(num_runs), desc=f"{strategy['name']}", leave=False):
            # Use different seed for each run to ensure independence
            run_seed = base_seed + run * 1000
            
            try:
                objective, solution = run_single_algorithm(
                    strategy=strategy,
                    data_path=data_path,
                    nc=nc,
                    nf=nf,
                    nt=nt,
                    seed=run_seed,
                    wires_per_feeder=wires_per_feeder,
                    max_iter=max_iter,
                    max_ils_iter=max_ils_iter
                )
                strategy_results.append(objective)
                
                # Store for boxplot
                all_results.append({
                    'Algorithm': strategy['name'],
                    'Objective': objective,
                    'Run': run + 1
                })
                
            except Exception as e:
                print(f"❌ Error in run {run+1} for {strategy['name']}: {e}")
                continue
        
        # Print statistics for this strategy
        if strategy_results:
            print(f"📊 {strategy['name']} Statistics:")
            print(f"   Mean: {np.mean(strategy_results):.6f}")
            print(f"   Std:  {np.std(strategy_results):.6f}")
            print(f"   Min:  {np.min(strategy_results):.6f}")
            print(f"   Max:  {np.max(strategy_results):.6f}")
            print(f"   Median: {np.median(strategy_results):.6f}")
    
    return all_results


def create_boxplot(results_data, save_path=None):
    """Create and display boxplot of algorithm performance."""
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results_data)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create boxplot using seaborn for better aesthetics
    ax = sns.boxplot(data=df, x='Algorithm', y='Objective', palette='Set2')
    
    # Customize the plot
    plt.title('Algorithm Performance Comparison\n(Lower is Better)', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Objective Function Value', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add statistics annotations
    algorithms = df['Algorithm'].unique()
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]['Objective']
        mean_val = alg_data.mean()
        median_val = alg_data.median()
        
        # Add mean marker
        plt.scatter(i, mean_val, color='red', s=100, marker='D', zorder=5, label='Mean' if i == 0 else "")
        
        # Add text annotation with mean value
        plt.text(i, mean_val + (alg_data.max() - alg_data.min()) * 0.1, 
                f'μ={mean_val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Boxplot saved to: {save_path}")
    
    # Show plot
    plt.show()
    
    return df


def run_statistical_analysis(results_data):
    """Perform statistical analysis on the results."""
    df = pd.DataFrame(results_data)
    
    print("\n" + "="*80)
    print("📈 STATISTICAL ANALYSIS")
    print("="*80)
    
    # Summary statistics
    summary = df.groupby('Algorithm')['Objective'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(6)
    
    print("\n📊 Summary Statistics:")
    print(summary)
    
    # Perform ANOVA test to check if there are significant differences
    try:
        from scipy import stats
        
        algorithms = df['Algorithm'].unique()
        groups = [df[df['Algorithm'] == alg]['Objective'].values for alg in algorithms]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        print(f"\n🧪 One-way ANOVA Test:")
        print(f"   F-statistic: {f_stat:.6f}")
        print(f"   P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("   ✅ Significant differences found between algorithms (p < 0.05)")
            
            # Perform pairwise t-tests
            print("\n🔍 Pairwise t-tests (Bonferroni corrected):")
            from itertools import combinations
            
            n_comparisons = len(algorithms) * (len(algorithms) - 1) // 2
            alpha_corrected = 0.05 / n_comparisons
            
            for alg1, alg2 in combinations(algorithms, 2):
                group1 = df[df['Algorithm'] == alg1]['Objective'].values
                group2 = df[df['Algorithm'] == alg2]['Objective'].values
                
                t_stat, p_val = stats.ttest_ind(group1, group2)
                significant = "✅" if p_val < alpha_corrected else "❌"
                
                print(f"   {alg1} vs {alg2}: t={t_stat:.3f}, p={p_val:.6f} {significant}")
        
        else:
            print("   ❌ No significant differences found between algorithms (p >= 0.05)")
            
    except ImportError:
        print("   ⚠️  scipy not available for statistical tests")
    
    return summary


def run_multiple_strategies_comparison(data_path, nc, nf, nt, seed, wires_per_feeder, max_iter, max_ils_iter):
    """Compare different search strategies."""
    print("\n" + "="*80)
    print("🔬 COMPARING MULTIPLE SEARCH STRATEGIES")
    print("="*80)
    
    strategies = [
        {
            "name": "Standard ILS",
            "acceptance": "better",
            "perturbation": lambda sol: random_perturbation(sol, num_changes=3, nw=nf * wires_per_feeder),
            "temperature": None
        },
        {
            "name": "Simulated Annealing ILS",
            "acceptance": "simulated_annealing", 
            "perturbation": lambda sol: random_perturbation(sol, num_changes=2, nw=nf * wires_per_feeder),
            "temperature": simulated_annealing_temperature
        }
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n🧪 Testing {strategy['name']}...")
        
        np.random.seed(seed)  # Ensure fair comparison
        nw = nf * wires_per_feeder
        
        generator = IdealGenerator(nc=nc, nf=nf, nt=nt, data_path=data_path)
        meter_supply, line_supply, _, topology = generator.generate(seed=seed)
        
        random_solution = np.random.randint(0, nw, size=nc)
        
        best_solution, best_objective = iterative_local_search(
            objective_function=objective_function_squared_sum,
            generate_neighbors=insert_neighborhood,
            perturbation=strategy["perturbation"],
            initial_solution=random_solution,
            nc=nc,
            nf=nf,
            meters=meter_supply,
            lines=line_supply,
            wires_per_feeder=wires_per_feeder,
            max_iterations=max_iter,
            max_ils_iterations=max_ils_iter // 3,  # Shorter runs for comparison
            verbose=False,  # Reduce output for comparison
            acceptance_criterion=strategy["acceptance"],
            temperature_schedule=strategy["temperature"],
            early_stopping_patience=None  # Disable early stopping
        )
        
        results.append({
            "strategy": strategy["name"],
            "objective": best_objective,
            "solution": best_solution
        })
    
    # Print comparison results
    print("\n" + "="*60)
    print("📊 STRATEGY COMPARISON RESULTS")
    print("="*60)
    
    results.sort(key=lambda x: x["objective"])
    
    for i, result in enumerate(results):
        rank_emoji = ["🥇", "🥈", "🥉"][min(i, 2)]
        print(f"{rank_emoji} {result['strategy']}: {result['objective']:.6f}")
    
    return results


if __name__ == "__main__":
    nc = 10  
    nf = 4     
    nt = 720
    wires_per_feeder = 3

    print(f"🚀 Running improved iterative local search with {nc} consumers, {nf} feeders, and {nt} time steps...\n")

    # Run single strategy test
    print("="*60)
    print("🔍 SINGLE RUN EXAMPLE")
    print("="*60)
    
    run_iterative_local_search(
        objective_function=objective_function_squared_sum,
        generate_neighbors=insert_neighborhood,
        perturbation=lambda sol: random_perturbation(sol, num_changes=3, nw=nf * wires_per_feeder),
        data_path="data/clean_data.csv",
        nc=nc,
        nf=nf,
        nt=nt,
        seed=42,
        wires_per_feeder=wires_per_feeder,
        max_iter=50,
        max_ils_iter=30
    )
    
    # Run multiple independent runs for statistical comparison
    print("\n" + "="*60)
    print("🎯 STATISTICAL COMPARISON")
    print("="*60)
    
    results_data = run_multiple_runs_comparison(
        data_path="data/clean_data.csv",
        nc=nc,
        nf=nf,
        nt=nt,
        base_seed=42,
        wires_per_feeder=wires_per_feeder,
        max_iter=30,  # Reduced for faster execution
        max_ils_iter=15,  # Reduced for faster execution
        num_runs=10  # Number of independent runs per algorithm
    )
    
    # Create boxplot
    if results_data:
        df = create_boxplot(results_data, save_path="algorithm_comparison_boxplot.png")
        
        # Perform statistical analysis
        summary = run_statistical_analysis(results_data)
        
        print(f"\n🎉 Analysis complete! Boxplot saved and {len(results_data)} total runs analyzed.")
    else:
        print("❌ No results collected for analysis.")