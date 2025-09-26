"""
VRPTW with Driver Preferences Solver

This module provides the main solver interface for optimizing vehicle routes
with driver personality preferences using Timefold Solver.
"""

from typing import List, Optional
import logging

# Configure detailed logging for Timefold
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('timefold_solver.log')
    ]
)

# Enable Timefold-specific logging
timefold_logger = logging.getLogger('timefold')
timefold_logger.setLevel(logging.INFO)

# Suppress matplotlib debug output
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# Suppress other common noisy loggers
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Set Java system properties for Timefold logging
import os
os.environ['JAVA_OPTS'] = '-Djava.util.logging.config.file=logging.properties'

# Alternative: Enable verbose Timefold logging via environment
os.environ['TIMEFOLD_SOLVER_LOG_LEVEL'] = 'DEBUG'

try:
    from timefold.solver import SolverFactory
    from timefold.solver.config import (
        SolverConfig, TerminationConfig, Duration, ScoreDirectorFactoryConfig
    )
    TIMEFOLD_AVAILABLE = True
except ImportError:
    TIMEFOLD_AVAILABLE = False
    logging.warning("Timefold not available. Solver will use fallback implementation.")

from .domain import VrptwSolution, Vehicle, Customer, Location, DriverProfile, OptimizationObjective
from .constraints import create_constraint_provider


class VrptwSolver:
    """Main solver class for VRPTW with configurable optimization objectives"""
    
    def __init__(self, solving_time_seconds: int = 30, objective: OptimizationObjective = OptimizationObjective.SHORTEST_DISTANCE, minimize_vehicles: bool = True, parallel: bool = False, enforce_capacity: bool = True, enforce_time_windows: bool = True):
        """
        Initialize the solver with configuration
        
        Args:
            solving_time_seconds: Maximum time to spend solving (default: 30s)
            objective: Optimization objective to use (default: SHORTEST_DISTANCE)
            minimize_vehicles: Whether to minimize vehicle usage as primary objective (default: True)
            parallel: Whether to enable parallel solving within Timefold (default: False)
            enforce_capacity: Whether to enforce vehicle capacity as hard constraint (default: True)
            enforce_time_windows: Whether to enforce time windows as hard constraint (default: True)
        """
        self.solving_time_seconds = solving_time_seconds
        self.objective = objective
        self.minimize_vehicles = minimize_vehicles
        self.parallel = parallel
        self.enforce_capacity = enforce_capacity
        self.enforce_time_windows = enforce_time_windows
        self.solver = self._create_solver() if TIMEFOLD_AVAILABLE else None
        
    def _create_solver(self):
        """Create and configure the Timefold solver with basic settings"""
        
        # Create constraint provider with current configuration
        constraint_provider = create_constraint_provider(
            objective=self.objective,
            minimize_vehicles=self.minimize_vehicles,
            enforce_capacity=self.enforce_capacity,
            enforce_time_windows=self.enforce_time_windows
        )
        
        # Create solver configuration with hierarchical scoring
        solver_config = SolverConfig(
            solution_class=VrptwSolution,
            entity_class_list=[Vehicle],
            score_director_factory_config=ScoreDirectorFactoryConfig(
                constraint_provider_function=constraint_provider
            ),
            termination_config=TerminationConfig(
                spent_limit=Duration(seconds=self.solving_time_seconds)
            )
        )
        
        # Enable parallel solving if requested (requires Timefold Enterprise)
        if self.parallel:
            import multiprocessing as mp
            thread_count = mp.cpu_count()
            solver_config.move_thread_count = str(thread_count)
            print(f"   🚀 Parallel solving enabled with {thread_count} threads")
        
        try:
            return SolverFactory.create(solver_config).build_solver()
        except Exception as e:
            if self.parallel and ("multithreaded solving" in str(e) or "RequiresEnterpriseError" in str(type(e))):
                print(f"   ⚠️  Parallel solving not available (requires Timefold Enterprise)")
                print(f"   📚 See: https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition")
                print(f"   🔄 Falling back to single-threaded solving")
                # Create new config without parallel settings
                solver_config = SolverConfig(
                    solution_class=VrptwSolution,
                    entity_class_list=[Vehicle],
                    score_director_factory_config=ScoreDirectorFactoryConfig(
                        constraint_provider_function=vrptw_constraints
                    ),
                    termination_config=TerminationConfig(
                        spent_limit=Duration(seconds=self.solving_time_seconds)
                    )
                )
                return SolverFactory.create(solver_config).build_solver()
            else:
                raise e
    
    def solve(self, problem: VrptwSolution) -> VrptwSolution:
        """
        Solve the VRPTW problem with the specified optimization objective
        
        Args:
            problem: The problem instance to solve
            
        Returns:
            The optimized solution
        """
        if not TIMEFOLD_AVAILABLE:
            logging.warning("Using fallback solver - optimization may be suboptimal")
            return self._fallback_solve(problem)
        
        logging.info(f"Starting optimization with {len(problem.vehicles)} vehicles "
                    f"and {len(problem.customers)} customers")
        
        # Enable detailed solver logging
        logger = logging.getLogger(__name__)
        logger.info(f"Solver configuration:")
        logger.info(f"  - Objective: {self.objective}")
        logger.info(f"  - Minimize vehicles: {self.minimize_vehicles}")
        logger.info(f"  - Time limit: {self.solving_time_seconds}s")
        logger.info(f"  - Parallel: {self.parallel}")
        
        solution = self.solver.solve(problem)
        
        logging.info(f"Optimization completed. Final score: {solution.score}")
        logging.info(f"Total distance: {solution.get_total_distance():.1f}")
        logging.info(f"Active vehicles: {len([v for v in solution.vehicles if len(v.visits) > 0])}")
        return solution
    
    def _fallback_solve(self, problem: VrptwSolution) -> VrptwSolution:
        """
        Fallback solver implementation using simple heuristics when Timefold is not available
        """
        logging.info("Using simple heuristic fallback solver")
        
        # Simple greedy assignment with personality matching
        unassigned_customers = problem.customers.copy()
        
        for vehicle in problem.vehicles:
            vehicle.visits = []
            
            # Sort customers by preference for this driver
            vehicle_customers = self._get_preferred_customers_for_vehicle(
                vehicle, unassigned_customers
            )
            
            # Assign customers while respecting capacity and route length limits
            current_capacity = 0
            for customer in vehicle_customers:
                if (current_capacity + customer.demand <= vehicle.capacity and 
                    len(vehicle.visits) < vehicle.driver.max_stops_per_route):
                    
                    vehicle.visits.append(customer)
                    current_capacity += customer.demand
                    unassigned_customers.remove(customer)
        
        # Simple score calculation for fallback
        problem.score = self._calculate_fallback_score(problem)
        
        return problem
    
    def _get_preferred_customers_for_vehicle(self, vehicle: Vehicle, customers: List[Customer]) -> List[Customer]:
        """Sort customers by simple heuristics for the given vehicle"""
        def preference_score(customer: Customer) -> float:
            score = 0.0
            
            # Distance penalty (closer is better)
            distance = vehicle.home_location.distance_to(customer.location)
            score -= distance * 0.1
            
            # Time window urgency (earlier due time gets priority)
            score -= customer.due_time * 0.01
            
            return score
        
        return sorted(customers, key=preference_score, reverse=True)
    
    def _calculate_fallback_score(self, problem: VrptwSolution):
        """Calculate a simple score for fallback implementation"""
        # This is a simplified score calculation
        total_distance = problem.get_total_distance()
        unassigned_count = len(problem.get_unassigned_customers())
        
        # Hard penalties
        hard_penalty = unassigned_count * 1000  # Unassigned customers
        
        # Soft penalties  
        soft_penalty = int(total_distance)
        
        # Create a simple score representation
        if TIMEFOLD_AVAILABLE:
            from timefold.solver.score import HardMediumSoftScore
            return HardMediumSoftScore.of(-hard_penalty, 0, -soft_penalty)
        else:
            return f"Hard: -{hard_penalty}, Medium: 0, Soft: -{soft_penalty}"


def create_sample_problem() -> VrptwSolution:
    """Create a sample problem for testing"""
    
    # Create depot
    depot = Location(id=0, x=50, y=50, name="Depot")
    
    # Create driver profiles
    drivers = [
        DriverProfile(id=1, name="Driver 1"),
        DriverProfile(id=2, name="Driver 2"),
        DriverProfile(id=3, name="Driver 3"),
    ]
    
    # Create vehicles
    vehicles = [
        Vehicle(id=i+1, driver=driver, home_location=depot, capacity=200)
        for i, driver in enumerate(drivers)
    ]
    
    # Create customers 
    customers = [
        Customer(id=1, location=Location(1, 20, 30), demand=25, ready_time=0, due_time=100),
        Customer(id=2, location=Location(2, 80, 70), demand=30, ready_time=0, due_time=120),
        Customer(id=3, location=Location(3, 40, 80), demand=20, ready_time=0, due_time=150),
        Customer(id=4, location=Location(4, 60, 20), demand=35, ready_time=0, due_time=110),
        Customer(id=5, location=Location(5, 30, 60), demand=40, ready_time=0, due_time=140),
    ]
    
    return VrptwSolution(
        depot=depot,
        customers=customers,
        vehicles=vehicles
    )


if __name__ == "__main__":
    # Test the solver
    logging.basicConfig(level=logging.INFO)
    
    print("Creating sample VRPTW problem with driver preferences...")
    problem = create_sample_problem()
    
    print(f"Problem created with {len(problem.customers)} customers and {len(problem.vehicles)} vehicles")
    
    solver = VrptwSolver(solving_time_seconds=10)
    solution = solver.solve(problem)
    
    print(f"\nSolution found!")
    print(f"Final score: {solution.score}")
    print(f"Total distance: {solution.get_total_distance():.1f}")
    print(f"Personality mismatches: {solution.get_total_personality_mismatches()}")
    
    for vehicle in solution.vehicles:
        print(f"\n{vehicle.driver.name} ({vehicle.driver.personality.value}):")
        print(f"  Visits: {len(vehicle.visits)} customers")
        print(f"  Total demand: {vehicle.get_total_demand()}")
        print(f"  Distance: {vehicle.get_total_distance():.1f}")
        if vehicle.visits:
            print(f"  Customer IDs: {[c.id for c in vehicle.visits]}")