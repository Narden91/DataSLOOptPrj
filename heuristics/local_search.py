import numpy as np
from tqdm import tqdm
import time
from typing import Callable, Tuple, Optional, Any
import inspect


def iterative_local_search(
    objective_function: Callable,
    generate_neighbors: Callable,
    perturbation: Callable,
    initial_solution: np.ndarray,
    nc: int,
    nf: int,
    meters: np.ndarray,
    lines: np.ndarray,
    wires_per_feeder: int = 3,
    max_iterations: int = 1000,
    max_ils_iterations: int = 10,
    verbose: bool = True,
    acceptance_criterion: str = "better",
    temperature_schedule: Optional[Callable] = None,
    early_stopping_patience: Optional[int] = 50,
    improvement_threshold: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Improved Iterative Local Search (ILS) algorithm with progress tracking and refined search strategies.
    
    Args:
        objective_function: Function to evaluate solution quality
        generate_neighbors: Function to generate neighboring solutions
        perturbation: Function to perturb/restart the search
        initial_solution: Starting solution
        nc: Number of consumers
        nf: Number of feeders/wires
        meters: Consumer consumption data
        lines: Feeder supply data
        wires_per_feeder: Number of wires per feeder
        max_iterations: Maximum local search iterations per ILS iteration
        max_ils_iterations: Maximum ILS iterations
        verbose: Whether to show progress bars and detailed output
        acceptance_criterion: "better", "simulated_annealing", or "threshold"
        temperature_schedule: Function for simulated annealing (if used)
        early_stopping_patience: Stop if no improvement for this many ILS iterations (None to disable)
        improvement_threshold: Minimum improvement to consider as progress
    
    Returns:
        Tuple of (best_solution, best_objective_value)
    """
    
    def local_search(solution: np.ndarray, max_iter: int, desc: str = "Local Search") -> Tuple[np.ndarray, float]:
        """Perform local search with first-improvement strategy."""
        current_solution = solution.copy()
        current_objective = objective_function(
            solution=current_solution,
            nc=nc,
            nf=nf,
            meters=meters,
            lines=lines,
            wires_per_feeder=wires_per_feeder
        )
        
        best_solution = current_solution.copy()
        best_objective = current_objective
        
        # Track consecutive iterations without improvement
        no_improvement_count = 0
        max_no_improvement = max(10, max_iter // 10)  # Adaptive early stopping
        
        # Progress bar for local search
        pbar = tqdm(range(max_iter), desc=desc, leave=False, disable=not verbose)
        
        for iteration in pbar:
            improved = False
            
            # Check the signature of generate_neighbors to call it correctly
            try:
                # Try to inspect the function signature
                sig = inspect.signature(generate_neighbors)
                params = list(sig.parameters.keys())
                
                # Call generate_neighbors with appropriate arguments
                if len(params) == 1:
                    # Simple case: only takes solution
                    neighbors = generate_neighbors(current_solution)
                elif 'nc' in params and 'nf' in params:
                    # Takes nc and nf parameters
                    neighbors = generate_neighbors(current_solution, nc=nc, nf=nf)
                elif 'nw' in params:
                    # Takes nw (number of wires) parameter
                    nw = nf * wires_per_feeder
                    neighbors = generate_neighbors(current_solution, nw=nw)
                else:
                    # Default case: try with just the solution
                    neighbors = generate_neighbors(current_solution)
                    
            except Exception as e:
                # Fallback: try with just the solution
                if verbose:
                    tqdm.write(f"Warning: Could not inspect neighbor function signature, using default call: {e}")
                neighbors = generate_neighbors(current_solution)
            
            # Generate all neighbors and evaluate them
            try:
                for neighbor in neighbors:
                    neighbor_objective = objective_function(
                        solution=neighbor,
                        nc=nc,
                        nf=nf,
                        meters=meters,
                        lines=lines,
                        wires_per_feeder=wires_per_feeder
                    )
                    
                    # Apply acceptance criterion
                    accept = False
                    if acceptance_criterion == "better":
                        accept = neighbor_objective < current_objective
                    elif acceptance_criterion == "simulated_annealing" and temperature_schedule:
                        temperature = temperature_schedule(iteration, max_iter)
                        if neighbor_objective < current_objective:
                            accept = True
                        else:
                            delta = neighbor_objective - current_objective
                            if temperature > 0:
                                probability = np.exp(-delta / temperature)
                                accept = np.random.random() < probability
                    elif acceptance_criterion == "threshold":
                        accept = neighbor_objective < current_objective + improvement_threshold
                    
                    if accept:
                        current_solution = neighbor.copy()
                        current_objective = neighbor_objective
                        improved = True
                        no_improvement_count = 0
                        
                        # Update best solution
                        if current_objective < best_objective:
                            best_solution = current_solution.copy()
                            best_objective = current_objective
                        
                        # First improvement: break to next iteration
                        break
                        
            except Exception as e:
                if verbose:
                    tqdm.write(f"Error during neighbor evaluation: {e}")
                break
            
            # Update progress bar
            pbar.set_postfix({
                'Best': f'{best_objective:.2e}',
                'Current': f'{current_objective:.2e}',
                'No Imp.': no_improvement_count
            })
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    if verbose:
                        tqdm.write(f"  Early stopping: no improvement for {no_improvement_count} iterations")
                    break
        
        pbar.close()
        return best_solution, best_objective
    
    # Initialize best solution
    best_solution = initial_solution.copy()
    best_objective = objective_function(
        solution=best_solution,
        nc=nc,
        nf=nf,
        meters=meters,
        lines=lines,
        wires_per_feeder=wires_per_feeder
    )
    
    if verbose:
        print(f"🎯 Initial objective: {best_objective:.6f}")
    
    # Track improvement history
    improvement_history = []
    last_improvement_iter = 0
    
    # Main ILS loop with progress tracking
    ils_pbar = tqdm(range(max_ils_iterations), desc="ILS Progress", disable=not verbose)
    
    for ils_iteration in ils_pbar:
        # Perform local search
        if ils_iteration == 0:
            search_solution = best_solution
        else:
            # Apply perturbation
            try:
                search_solution = perturbation(best_solution)
            except Exception as e:
                if verbose:
                    tqdm.write(f"Error in perturbation: {e}, using random solution")
                nw = nf * wires_per_feeder
                search_solution = np.random.randint(0, nw, size=nc)
        
        current_solution, current_objective = local_search(
            solution=search_solution,
            max_iter=max_iterations,
            desc=f"ILS {ils_iteration+1}/{max_ils_iterations}"
        )
        
        # Check for improvement
        improvement = best_objective - current_objective
        improvement_history.append(improvement)
        
        # Update best solution if improved
        if current_objective < best_objective:
            improvement_ratio = improvement / abs(best_objective) if abs(best_objective) > 1e-10 else float('inf')
            best_solution = current_solution.copy()
            best_objective = current_objective
            last_improvement_iter = ils_iteration
            
            if verbose:
                tqdm.write(f"🎉 New best at ILS {ils_iteration+1}: {best_objective:.6f} "
                          f"(improvement: {improvement:.2e}, {improvement_ratio:.2%})")
        
        # Update progress bar
        recent_improvements = sum(1 for imp in improvement_history[-10:] if imp > improvement_threshold)
        ils_pbar.set_postfix({
            'Best': f'{best_objective:.2e}',
            'Current': f'{current_objective:.2e}',
            'Last Imp.': ils_iteration - last_improvement_iter,
            'Recent': f'{recent_improvements}/10'
        })
        
        # Early stopping based on patience (only if patience is not None)
        if early_stopping_patience is not None:
            if ils_iteration - last_improvement_iter >= early_stopping_patience:
                if verbose:
                    tqdm.write(f"🛑 Early stopping: no improvement for {early_stopping_patience} ILS iterations")
                break
        
        # Adaptive behavior based on recent performance
        if ils_iteration > 0 and ils_iteration % 20 == 0:
            recent_avg_improvement = np.mean(improvement_history[-20:])
            if recent_avg_improvement < improvement_threshold:
                if verbose:
                    tqdm.write(f"🔄 Low recent improvement ({recent_avg_improvement:.2e})")
    
    ils_pbar.close()
    
    if verbose:
        initial_obj = objective_function(initial_solution, nc, nf, meters, lines, wires_per_feeder)
        total_improvement = initial_obj - best_objective
        improvement_percentage = (total_improvement / initial_obj) * 100 if initial_obj > 0 else 0
        print(f"\n📈 Total improvement: {total_improvement:.6f} ({improvement_percentage:.2f}%)")
        print(f"🏆 Final best objective: {best_objective:.6f}")
    
    return best_solution, best_objective


def simulated_annealing_temperature(iteration: int, max_iterations: int, 
                                  initial_temp: float = 1000.0, 
                                  final_temp: float = 0.01) -> float:
    """
    Exponential cooling schedule for simulated annealing.
    
    Args:
        iteration: Current iteration
        max_iterations: Maximum number of iterations
        initial_temp: Starting temperature
        final_temp: Final temperature
    
    Returns:
        Current temperature
    """
    if max_iterations <= 1:
        return final_temp
    
    alpha = (final_temp / initial_temp) ** (1.0 / (max_iterations - 1))
    return initial_temp * (alpha ** iteration)


def adaptive_perturbation(solution: np.ndarray, base_changes: int = 3, 
                         performance_history: Optional[list] = None,
                         nw: int = None) -> np.ndarray:
    """
    Adaptive perturbation that adjusts strength based on recent performance.
    
    Args:
        solution: Current solution to perturb
        base_changes: Base number of changes to make
        performance_history: List of recent improvement values
        nw: Number of wires (for random_perturbation)
    
    Returns:
        Perturbed solution
    """
    from heuristics.utils import random_perturbation
    
    # Adjust perturbation strength based on recent performance
    num_changes = base_changes
    
    if performance_history and len(performance_history) >= 5:
        recent_improvements = performance_history[-5:]
        avg_improvement = np.mean(recent_improvements)
        
        # If little improvement, increase perturbation strength
        if avg_improvement < 1e-6:
            num_changes = min(base_changes * 3, len(solution) // 4)
        elif avg_improvement < 1e-4:
            num_changes = base_changes * 2
    
    return random_perturbation(solution, num_changes=num_changes, nw=nw)