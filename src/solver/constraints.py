"""
VRPTW with Driver Preferences Constraint Provider

This module defines the constraints for optimizing vehicle routes
with driver personality preferences using Timefold Solver.
"""

try:
    from timefold.solver.score import constraint_provider, HardSoftScore, HardMediumSoftScore, Joiners
except ImportError:
    # Fallback for when Timefold is not available
    def constraint_provider(cls): return cls
    class HardSoftScore:
        ONE_HARD = 1
        ONE_SOFT = 1
    class HardMediumSoftScore:
        ONE_HARD = 1
        ONE_MEDIUM = 1
        ONE_SOFT = 1
    class Joiners:
        @staticmethod
        def equal(*args): return None

from .domain import Vehicle, Customer, OptimizationObjective
import logging

# Set up constraint logging
constraint_logger = logging.getLogger('constraints')


# Remove all global variables - pass parameters properly instead

def create_constraint_provider(objective: OptimizationObjective, minimize_vehicles: bool = True, 
                              enforce_capacity: bool = True, enforce_time_windows: bool = True):
    """Factory function to create constraint provider with specific configuration"""
    
    @constraint_provider
    def vrptw_constraints(constraint_factory):
        """Define all constraints for VRPTW with specific configuration"""
        constraints = []
        
        # Add objective-specific constraint based on the selected optimization objective
        if objective == OptimizationObjective.CLUSTERED:
            constraints.append(clustering_constraint(constraint_factory))
        elif objective == OptimizationObjective.SHORTEST_DISTANCE:
            constraints.append(distance_constraint(constraint_factory))
        elif objective == OptimizationObjective.EVEN_STOPS:
            constraints.append(even_stops_constraint(constraint_factory))
        elif objective == OptimizationObjective.RADIAL:
            constraints.append(radial_constraint(constraint_factory))
        
        # Add hard constraints if enabled
        if enforce_capacity:
            constraints.append(vehicle_capacity_constraint(constraint_factory))
        
        if enforce_time_windows:
            constraints.append(time_window_constraint(constraint_factory))
        
        # Add vehicle minimization constraint if enabled
        if minimize_vehicles:
            constraints.append(minimize_vehicles_constraint(constraint_factory))
        
        return constraints
    
    return vrptw_constraints


def vehicle_capacity_constraint(constraint_factory):
    """Hard constraint: Vehicle capacity must not be exceeded"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: vehicle.get_total_demand() > vehicle.capacity)
            .penalize(HardMediumSoftScore.ONE_HARD,
                     lambda vehicle: vehicle.get_total_demand() - vehicle.capacity)
            .as_constraint("Vehicle capacity"))


def time_window_constraint(constraint_factory):
    """Hard constraint: All time windows must be respected"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: vehicle.violates_time_windows())
            .penalize(HardMediumSoftScore.ONE_HARD,
                     lambda vehicle: calculate_time_window_violation(vehicle))
            .as_constraint("Time windows"))


def objective_optimization_constraint(constraint_factory, objective: OptimizationObjective):
    """Soft constraint: Optimize routes based on the selected objective (SOFT level)"""
    # Use string representation instead of .value for Java interop
    objective_name = str(objective).split('.')[-1].lower()  # Get "clustered" from "OptimizationObjective.CLUSTERED"
    
    # Create the penalty function that captures the objective
    def penalty_function(vehicle):
        return calculate_objective_penalty(vehicle, objective)
    
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only penalize vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT, penalty_function)
            .as_constraint(f"Optimize for {objective_name}"))


def clustering_constraint(constraint_factory):
    """Direct clustering constraint - minimizes geographical spread of routes"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) >= 2)  # Only vehicles with 2+ customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: calculate_cluster_penalty(vehicle))
            .as_constraint("Clustering penalty"))


def distance_constraint(constraint_factory):
    """Distance constraint - minimizes total travel distance"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: calculate_distance_penalty(vehicle))
            .as_constraint("Distance penalty"))


def even_stops_constraint(constraint_factory):
    """Even stops constraint - balances stops across vehicles"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: calculate_even_stops_penalty(vehicle))
            .as_constraint("Even stops penalty"))


def radial_constraint(constraint_factory):
    """Radial constraint - optimizes for depot-centric patterns"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: calculate_radial_penalty(vehicle))
            .as_constraint("Radial penalty"))


def debug_clustering_constraint(constraint_factory):
    """Debug constraint to test clustering penalty application"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: len(vehicle.visits) * 1000)  # 1000 per customer
            .as_constraint("Debug clustering"))


def simple_test_constraint(constraint_factory):
    """Simple test constraint to verify constraint system is working"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only vehicles with customers
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: 100)  # Fixed penalty of 100 per vehicle with customers
            .as_constraint("Test constraint"))


def minimize_vehicles_constraint(constraint_factory):
    """Medium constraint: Minimize the number of vehicles used (PRIMARY objective at MEDIUM level)"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only count vehicles with customers
            .penalize(HardMediumSoftScore.ONE_MEDIUM,
                     lambda vehicle: 1)  # Simple penalty - each vehicle used gets penalty 1
            .as_constraint("Minimize vehicles"))


# Helper functions for constraint calculations

def calculate_time_window_violation(vehicle: Vehicle) -> int:
    """Calculate the severity of time window violations"""
    if len(vehicle.visits) == 0:
        return 0
        
    violation_penalty = 0
    current_time = 0.0
    current_location = vehicle.home_location
    
    for customer in vehicle.visits:
        # Calculate arrival time
        travel_time = current_location.distance_to(customer.location)
        arrival_time = current_time + travel_time
        
        # Penalize late arrival
        if arrival_time > customer.due_time:
            violation_penalty += int((arrival_time - customer.due_time) * 10)
        
        # Update for next customer
        service_start = max(arrival_time, customer.ready_time)
        current_time = service_start + customer.service_time
        current_location = customer.location
    
    return violation_penalty


def calculate_objective_penalty(vehicle: Vehicle, objective: OptimizationObjective) -> int:
    """Calculate penalty based on how well the route matches the optimization objective"""
    if len(vehicle.visits) == 0:
        return 0
    
    penalty = 0
    
    if objective == OptimizationObjective.CLUSTERED:
        penalty = calculate_cluster_penalty(vehicle)
    elif objective == OptimizationObjective.SHORTEST_DISTANCE:
        penalty = calculate_distance_penalty(vehicle)
    elif objective == OptimizationObjective.EVEN_STOPS:
        penalty = calculate_even_stops_penalty(vehicle)
    elif objective == OptimizationObjective.RADIAL:
        penalty = calculate_radial_penalty(vehicle)
    
    return penalty


def calculate_cluster_penalty(vehicle: Vehicle) -> int:
    """Optimized clustering penalty - minimizes geographical spread"""
    visits = vehicle.visits
    if len(visits) < 2:
        return 0
    
    # Fast centroid calculation - avoid list comprehension
    n = len(visits)
    cx = sum(c.location.x for c in visits) / n
    cy = sum(c.location.y for c in visits) / n
    
    # Use squared distance (avoid expensive sqrt) and simpler penalty
    penalty = 0
    for customer in visits:
        # Squared Euclidean distance (no sqrt needed)
        dx = customer.location.x - cx
        dy = customer.location.y - cy
        squared_distance = dx * dx + dy * dy
        
        # Linear penalty based on squared distance (much faster than distance^3)
        penalty += int(squared_distance * 10)  # Scale factor for meaningful penalties
    
    return penalty


def calculate_distance_penalty(vehicle: Vehicle) -> int:
    """Penalty for SHORTEST_DISTANCE drivers - just the total distance"""
    return int(vehicle.get_total_distance())


def calculate_even_stops_penalty(vehicle: Vehicle) -> int:
    """Penalty for EVEN_STOPS drivers - penalize imbalanced routes"""
    # This will be calculated globally by comparing all vehicles
    # For now, slightly penalize very long or very short routes
    num_stops = len(vehicle.visits)
    
    # Ideal range is 8-12 stops
    if num_stops < 5:
        return (5 - num_stops) * 50  # Penalty for too few stops
    elif num_stops > 15:
        return (num_stops - 15) * 50  # Penalty for too many stops
    
    return 0


def calculate_radial_penalty(vehicle: Vehicle) -> int:
    """Penalty for RADIAL drivers - penalize non-radial patterns"""
    if len(vehicle.visits) < 2:
        return 0
    
    depot = vehicle.home_location
    penalty = 0
    
    # Penalize routes that don't follow a radial pattern
    # A good radial route goes out from depot and returns
    for i in range(len(vehicle.visits) - 1):
        current = vehicle.visits[i]
        next_customer = vehicle.visits[i + 1]
        
        # Calculate distances from depot
        current_dist_from_depot = depot.distance_to(current.location)
        next_dist_from_depot = depot.distance_to(next_customer.location)
        
        # Penalize if we're moving towards depot in the middle of the route
        if i < len(vehicle.visits) // 2 and next_dist_from_depot < current_dist_from_depot:
            penalty += 100  # Moving inward too early
        # Penalize if we're moving away from depot at the end of the route
        elif i >= len(vehicle.visits) // 2 and next_dist_from_depot > current_dist_from_depot:
            penalty += 100  # Moving outward too late
    
    return penalty




