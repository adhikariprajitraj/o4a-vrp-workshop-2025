"""
VRPTW with Driver Preferences Domain Model

This module defines the domain objects for Vehicle Routing Problem with Time Windows
and Driver Preferences using Timefold Solver.
"""

from dataclasses import dataclass, field
from typing import Annotated, Optional, List
from enum import Enum

try:
    from timefold.solver.domain import (
        planning_entity, 
        planning_solution,
        PlanningId, 
        PlanningVariable,
        PlanningListVariable,
        PlanningScore,
        PlanningEntityCollectionProperty,
        ValueRangeProvider
    )
    from timefold.solver.score import HardMediumSoftScore
except ImportError:
    # Fallback for when Timefold is not available
    def planning_entity(cls): return cls
    def planning_solution(cls): return cls
    def PlanningId(): return None
    def PlanningVariable(): return None
    def PlanningListVariable(): return None
    def PlanningScore(): return None
    def PlanningEntityCollectionProperty(): return None
    def ValueRangeProvider(): return None
    class HardMediumSoftScore:
        ZERO = None




@dataclass(frozen=True)
class Location:
    """Represents a location with coordinates and properties"""
    id: int
    x: float
    y: float
    name: str = ""
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate Euclidean distance to another location"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class Customer:
    """Represents a customer with delivery requirements and time windows"""
    id: Annotated[int, PlanningId]
    location: Location
    demand: int
    ready_time: float
    due_time: float
    service_time: float = 0.0
    is_premium: bool = False


@dataclass
class DriverProfile:
    """Represents a driver with basic information (personality-agnostic)"""
    id: int
    name: str
    
    # Basic driver constraints
    max_stops_per_route: int = 15
    


@planning_entity
@dataclass
class Vehicle:
    """Represents a vehicle with driver and route assignments"""
    id: Annotated[int, PlanningId]
    driver: DriverProfile
    home_location: Location
    capacity: int
    
    # This is what Timefold will optimize - the sequence of customer visits
    visits: Annotated[List[Customer], PlanningListVariable] = field(default_factory=list)
    
    def get_total_demand(self) -> int:
        """Calculate total demand for all visits"""
        return sum(customer.demand for customer in self.visits)
    
    def get_total_distance(self) -> float:
        """Calculate total route distance"""
        if len(self.visits) == 0:
            return 0.0
            
        total_distance = 0.0
        current_location = self.home_location
        
        for customer in self.visits:
            total_distance += current_location.distance_to(customer.location)
            current_location = customer.location
            
        # Return to depot
        total_distance += current_location.distance_to(self.home_location)
        return total_distance
    
    def get_personality_mismatch_count(self) -> int:
        """Count how many customers don't match driver personality"""
        mismatch_count = 0
        for customer in self.visits:
            preferred = customer.preferred_driver_personality
            if preferred is not None and preferred != self.driver.personality:
                mismatch_count += 1
        return mismatch_count
    
    def violates_time_windows(self) -> bool:
        """Check if route violates any time windows"""
        if len(self.visits) == 0:
            return False
            
        current_time = 0.0
        current_location = self.home_location
        
        for customer in self.visits:
            # Travel time (simplified as distance)
            travel_time = current_location.distance_to(customer.location)
            arrival_time = current_time + travel_time
            
            # Check time window violation
            if arrival_time > customer.due_time:
                return True
                
            # Wait if arrived early
            service_start = max(arrival_time, customer.ready_time)
            current_time = service_start + customer.service_time
            current_location = customer.location
            
        return False


@planning_solution
@dataclass
class VrptwSolution:
    """The planning solution containing all vehicles, customers, and constraints"""
    
    # Problem facts (don't change during optimization)
    depot: Location
    customers: Annotated[List[Customer], ValueRangeProvider]
    vehicles: Annotated[List[Vehicle], PlanningEntityCollectionProperty]
    
    # Planning score (Timefold will optimize this)
    score: Annotated[Optional[HardMediumSoftScore], PlanningScore] = field(default=None)
    
    def get_total_distance(self) -> float:
        """Calculate total distance for all vehicles"""
        return sum(vehicle.get_total_distance() for vehicle in self.vehicles)
    
    def get_unassigned_customers(self) -> List[Customer]:
        """Get list of customers not assigned to any vehicle"""
        assigned_customers = set()
        for vehicle in self.vehicles:
            assigned_customers.update(vehicle.visits)
        
        return [customer for customer in self.customers if customer not in assigned_customers]