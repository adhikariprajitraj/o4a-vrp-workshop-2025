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

from .domain import Vehicle, Customer
import logging

# Set up constraint logging
constraint_logger = logging.getLogger('constraints')


# Auto-Loading Constraint System
# =============================
#
# To add your own custom constraint, simply:
# 1. Define your constraint function (see examples below)
# 2. Add it to the ACTIVE_CONSTRAINTS list at the bottom of this file
# 3. The solver will automatically load and use all constraints in the list
#
# Example custom constraint:
#
# def my_custom_constraint(constraint_factory):
#     """Example: Penalize routes with more than 10 stops"""
#     return (constraint_factory
#             .for_each(Vehicle)
#             .filter(lambda vehicle: len(vehicle.visits) > 10)
#             .penalize(HardMediumSoftScore.ONE_SOFT,
#                      lambda vehicle: (len(vehicle.visits) - 10) * 100)
#             .as_constraint("Too many stops penalty"))
#
# Available constraint levels:
# - HardMediumSoftScore.ONE_HARD: Must be satisfied (violation = infeasible solution)
# - HardMediumSoftScore.ONE_MEDIUM: Should be minimized (higher priority than soft)
# - HardMediumSoftScore.ONE_SOFT: Nice to have (lowest priority)
#
# Useful Vehicle properties:
# - vehicle.visits: List of Customer objects assigned to this vehicle
# - vehicle.capacity: Maximum capacity of the vehicle
# - vehicle.home_location: Starting/ending depot location
# - vehicle.get_total_demand(): Total demand of all assigned customers
# - vehicle.get_total_distance(): Total travel distance for the route
#
# Useful Customer properties:
# - customer.location: Location object with x, y coordinates
# - customer.demand: Resource demand (weight, volume, etc.)
# - customer.ready_time: Earliest service time
# - customer.due_time: Latest service time
# - customer.service_time: Time needed to serve this customer

def create_constraint_provider(constraints_list: list = None):
    """Factory function to create constraint provider with a list of constraints

    Args:
        constraints_list: List of constraint functions to apply.
                         If None, uses the ACTIVE_CONSTRAINTS list from this module
    """
    if constraints_list is None:
        constraints_list = ACTIVE_CONSTRAINTS

    @constraint_provider
    def vrptw_constraints(constraint_factory):
        """Define all constraints for VRPTW"""
        constraints = []

        # Apply all constraints from the list
        for constraint_func in constraints_list:
            constraints.append(constraint_func(constraint_factory))

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


def minimize_vehicles_constraint(constraint_factory):
    """Medium constraint: Minimize the number of vehicles used (PRIMARY objective at MEDIUM level)"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)  # Only count vehicles with customers
            .penalize(HardMediumSoftScore.ONE_MEDIUM,
                     lambda vehicle: 1)  # Simple penalty - each vehicle used gets penalty 1
            .as_constraint("Minimize vehicles"))



def minimize_total_distance_constraint(constraint_factory):
    """Example: Minimize total travel distance across all routes"""
    return (constraint_factory
            .for_each(Vehicle)
            .filter(lambda vehicle: len(vehicle.visits) > 0)
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     lambda vehicle: int(vehicle.get_total_distance()))
            .as_constraint("Minimize total distance"))



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


def example_custom_constraint(constraint_factory):
    """Template: Penalize routes with more than 8 stops (modify as needed)"""
    return (constraint_factory
            .for_each(Vehicle)
            # Your condition here
            .filter(lambda vehicle: len(vehicle.visits) > 8)
            .penalize(HardMediumSoftScore.ONE_SOFT,
                      # Your penalty here
                      # You can also pass in your own function here that is defined in this file
                     lambda vehicle: (len(vehicle.visits) - 8) * 50)
            .as_constraint("Too many stops"))  # Your constraint name here


# =============================================================================
# ACTIVE CONSTRAINTS LIST
# =============================================================================
# Add or remove constraint functions from this list to customize the solver.
# The solver will automatically use all constraints listed here.

ACTIVE_CONSTRAINTS = [
    vehicle_capacity_constraint,
    time_window_constraint,
    minimize_vehicles_constraint,

    # Example template constraint (uncomment to enable)
    # minimize_total_distance_constraint,

    # Add your custom constraints here:
    # your_custom_constraint,
]





