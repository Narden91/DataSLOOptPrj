import sys
sys.dont_write_bytecode = True

from feeder_mapping.generator import IdealGenerator
from feeder_mapping.objective import objective_function_squared_sum
from heuristics.neighborhoods import insert_neighborhood
from heuristics.local_search import local_search
import numpy as np


def run_local_search(objective_function, generate_neighbors, data_path, nc, nf, nt, seed, wires_per_feeder, max_iter):
    np.random.seed(seed)

    nw = nf * wires_per_feeder
    print(f"🔧 Initializing IdealGenerator with {nc} consumers, {nf} feeders, and {nt} time steps...")

    generator = IdealGenerator(nc=nc, nf=nw, nt=nt, data_path=data_path)
    meter_supply, line_supply, _, topology = generator.generate(seed=seed)

    print("📊 Meter Supply Shape:", meter_supply.shape)
    print("🔌 Line Supply Shape:", line_supply.shape)
    print("🧩 Ground Truth Feeder Topology:", topology)
    print(f"🧮 System has {nw} wires ({wires_per_feeder} per feeder)\n")

    # Step 1: Random initial solution
    random_solution = np.random.randint(0, nw, size=nc)
    current_score = objective_function(
        solution=random_solution,
        nc=nc,
        nf=nw,
        meters=meter_supply,
        lines=line_supply,
        wires_per_feeder=wires_per_feeder
    )

    print(f"🎯 Initial score: {current_score:.4f}")

    best_solution, best_objective = local_search(
        objective_function,
        random_solution,
        generate_neighbors,
        nc,
        nf,
        meter_supply,
        line_supply,
        wires_per_feeder,
        max_iterations=max_iter,
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

    print(f"🚀 Running local search with {nc} consumers, {nf} feeders, and {nt} time steps...\n")
    run_local_search(
        objective_function=objective_function_squared_sum,
        generate_neighbors=insert_neighborhood,
        data_path="../data/clean_data.csv",
        nc=nc,
        nf=nf,
        nt=nt,
        seed=42,
        wires_per_feeder=wires_per_feeder,
        max_iter=100
    )
