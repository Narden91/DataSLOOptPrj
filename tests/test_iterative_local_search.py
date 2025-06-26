import sys
sys.dont_write_bytecode = True

from feeder_mapping.generator import IdealGenerator
from feeder_mapping.objective import objective_function_squared_sum
from heuristics.neighborhoods import insert_neighborhood
from heuristics.local_search import iterative_local_search
from heuristics.utils import random_perturbation
import numpy as np

def run_iterative_local_search(objective_function, generate_neighbors, perturbation, data_path,
                               nc, nf, nt, seed, wires_per_feeder, max_iter, max_ils_iter):
    np.random.seed(seed)

    nw = nf * wires_per_feeder
    print(f"🔧 Initializing IdealGenerator with {nc} consumers, {nf} feeders, and {nt} time steps...")

    generator = IdealGenerator(nc=nc, nf=nw, nt=nt, data_path=data_path)
    meter_supply, line_supply, _, topology = generator.generate(seed=seed)

    print("📊 Meter Supply Shape:", meter_supply.shape)
    print("🔌 Line Supply Shape:", line_supply.shape)
    print("🧩 Ground Truth Feeder Topology:", topology)
    print(f"🧮 System has {nw} wires ({wires_per_feeder} per feeder)\n")

    random_solution = np.random.randint(0, nw, size=nc)
    initial_score = objective_function(
        solution=random_solution,
        nc=nc,
        nf=nw,
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder
    )
    print(f"🎯 Initial score: {initial_score:.4f}")

    best_solution, best_objective = iterative_local_search(
        objective_function=objective_function,
        generate_neighbors=generate_neighbors,
        perturbation=perturbation,
        initial_solution=random_solution,
        nc=nc,
        nf=nw,
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder,
        max_iterations=max_iter,
        max_ils_iterations=max_ils_iter,
        verbose=True
    )

    print("\n🏁 Final solution (first 10):", best_solution[:10])
    print(f"🏆 Final objective score: {best_objective:.4f}")
    return best_solution, best_objective


if __name__ == "__main__":
    nc = 10
    nf = 2
    nt = 720
    wires_per_feeder = 3

    print(f"🚀 Running iterative local search with {nc} consumers, {nf} feeders, and {nt} time steps...\n")

    run_iterative_local_search(
        objective_function=objective_function_squared_sum,
        generate_neighbors=insert_neighborhood,
        perturbation=lambda sol: random_perturbation(sol, num_changes=3, nw=nf * wires_per_feeder),
        data_path="../data/clean_data.csv",
        nc=nc,
        nf=nf,
        nt=nt,
        seed=40,
        wires_per_feeder=wires_per_feeder,
        max_iter=100,
        max_ils_iter=10000
    )
