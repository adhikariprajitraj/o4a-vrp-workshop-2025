"""
Integration between Homberger datasets and VRPTW solver with driver preferences

This module converts Homberger dataset format to our VRPTW domain model
and adds intelligent driver preference assignments.
"""

from typing import List, Dict

from src.io.dataloaders import HombergerDataLoader
from src.solver.domain import (
    VrptwSolution, Vehicle, Customer, Location, 
    DriverProfile, OptimizationObjective
)


class HombergerVRPTWAdapter:
    """Adapts Homberger datasets to VRPTW problems"""
    
    def __init__(self):
        """Initialize the adapter"""
        pass
    
    def adapt_dataset(self, dataset_path: str, num_vehicles: int = None) -> VrptwSolution:
        """
        Adapt a Homberger dataset to VRPTW problem
        
        Args:
            dataset_path: Path to the Homberger dataset file
            num_vehicles: Number of vehicles to create (if None, reads from file)
            
        Returns:
            VrptwSolution problem instance
        """
        # Load the Homberger dataset
        loader = HombergerDataLoader(dataset_path)
        df = loader.load()
        
        # Get vehicle information from the file
        with open(dataset_path, 'r') as f:
            content = f.read()
        dataset_info = loader._parse_homberger_format(content)
        
        # Use vehicle count from file if not specified
        if num_vehicles is None:
            num_vehicles = dataset_info['vehicle_num']
        vehicle_capacity = dataset_info['capacity']
        
        df_pandas = df.to_pandas()
        
        # Extract depot and customers
        depot_row = df_pandas[df_pandas['customer_id'] == 0].iloc[0]
        customer_rows = df_pandas[df_pandas['customer_id'] != 0]
        
        # Create depot location
        depot = Location(
            id=0,
            x=float(depot_row['x']),
            y=float(depot_row['y']),
            name="Depot"
        )
        
        # Create customers
        customers = self._create_customers(customer_rows)
        
        # Create simple driver profiles and vehicles  
        drivers = self._create_driver_profiles(num_vehicles)
        vehicles = self._create_vehicles(drivers, depot, vehicle_capacity)
        
        return VrptwSolution(
            depot=depot,
            customers=customers,
            vehicles=vehicles
        )
    
    def _create_customers(self, customer_rows) -> List[Customer]:
        """Create customers from dataset"""
        customers = []
        
        for _, row in customer_rows.iterrows():
            location = Location(
                id=int(row['customer_id']),
                x=float(row['x']),
                y=float(row['y']),
                name=f"Customer {int(row['customer_id'])}"
            )
            
            customer = Customer(
                id=int(row['customer_id']),
                location=location,
                demand=int(row['demand']),
                ready_time=float(row['ready_time']),
                due_time=float(row['due_time']),
                service_time=float(row.get('service_time', 10))  # Default service time
            )
            
            customers.append(customer)
        
        return customers
    
    
    def _create_driver_profiles(self, num_vehicles: int) -> List[DriverProfile]:
        """Create simple driver profiles"""
        drivers = []
        
        for i in range(num_vehicles):
            driver = DriverProfile(
                id=i + 1,
                name=f"Driver {i + 1}"
            )
            drivers.append(driver)
        
        return drivers
    
    def _create_vehicles(self, drivers: List[DriverProfile], depot: Location, vehicle_capacity: int) -> List[Vehicle]:
        """Create vehicles with the given drivers"""
        vehicles = []
        
        for i, driver in enumerate(drivers):
            vehicle = Vehicle(
                id=i + 1,
                driver=driver,
                home_location=depot,
                capacity=vehicle_capacity
            )
            vehicles.append(vehicle)
        
        return vehicles


def generate_optimization_report(solution: VrptwSolution, objective: OptimizationObjective) -> Dict:
    """Generate a detailed report of optimization results"""
    
    total_customers = len(solution.customers)
    total_assigned = sum(len(v.visits) for v in solution.vehicles)
    
    # Import here to avoid circular import
    from ..solver.constraints import calculate_objective_penalty
    
    total_penalty = 0
    route_scores = []
    
    for vehicle in solution.vehicles:
        if len(vehicle.visits) > 0:
            penalty = calculate_objective_penalty(vehicle, objective)
            score = max(0, 1000 - penalty)  # Convert penalty to score
            route_scores.append(score)
            total_penalty += penalty
    
    avg_score = sum(route_scores) / len(route_scores) if route_scores else 0
    
    return {
        'objective': objective.value,
        'total_customers': total_customers,
        'total_assigned': total_assigned,
        'unassigned': len(solution.get_unassigned_customers()),
        'total_distance': solution.get_total_distance(),
        'total_penalty': total_penalty,
        'average_route_score': avg_score,
        'active_vehicles': len([v for v in solution.vehicles if len(v.visits) > 0])
    }


def create_comparison_problems(dataset_path: str) -> Dict[str, VrptwSolution]:
    """Create problems for comparing different objectives"""
    
    adapter = HombergerVRPTWAdapter()
    
    # Create identical problems - differences will be in the objective used during solving
    problems = {
        'problem1': adapter.adapt_dataset(dataset_path, num_vehicles=10),
        'problem2': adapter.adapt_dataset(dataset_path, num_vehicles=10)
    }
    
    return problems


if __name__ == "__main__":
    # Test the adapter
    dataset_path = '../../data/homberger_200_customer_instances/C1_2_1.TXT'
    
    print("Adapting Homberger dataset to VRPTW problem...")
    adapter = HombergerVRPTWAdapter()
    problem = adapter.adapt_dataset(dataset_path, num_vehicles=3)
    
    print(f"Problem created:")
    print(f"  Depot: {problem.depot.name} at ({problem.depot.x}, {problem.depot.y})")
    print(f"  Customers: {len(problem.customers)}")
    print(f"  Vehicles: {len(problem.vehicles)}")
    
    print(f"\nDriver profiles:")
    for vehicle in problem.vehicles:
        driver = vehicle.driver
        print(f"  {driver.name}: {driver.personality.value}")
        print(f"    Max stops: {driver.max_stops_per_route}")
        print(f"    Service rating: {driver.customer_service_rating}")
    
    print(f"\nCustomer preference distribution:")
    high_value = sum(1 for c in problem.customers if c.is_high_value == 1)
    time_critical = sum(1 for c in problem.customers if c.is_time_critical == 1)
    complex_delivery = sum(1 for c in problem.customers if c.is_complex_delivery == 1)
    
    print(f"  High value customers: {high_value}")
    print(f"  Time critical customers: {time_critical}")
    print(f"  Complex delivery customers: {complex_delivery}")
    print(f"  Regular customers: {len(problem.customers) - high_value - time_critical - complex_delivery}")