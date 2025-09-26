import matplotlib.pyplot as plt
import numpy as np
from src.io.dataloaders import HombergerDataLoader
import os
import polars as pl
import random
from typing import Optional

def generate_fake_routes(df: pl.DataFrame, num_vehicles: int = None) -> pl.DataFrame:
    """
    Generate fake but realistic routes for a given dataset.
    
    Args:
        df (pl.DataFrame): The dataset containing customer locations and time windows
        num_vehicles (int, optional): Number of vehicles. If None, estimated from data
    
    Returns:
        pl.DataFrame: Routes dataframe with columns: vehicle_id, customer_id, sequence, arrival_time
    """
    # Convert to pandas for easier manipulation
    df_pandas = df.to_pandas()
    
    # Separate depot and customers
    depot = df_pandas[df_pandas['customer_id'] == 0].iloc[0]
    customers = df_pandas[df_pandas['customer_id'] != 0].copy()
    
    # Estimate number of vehicles if not provided (based on total demand and typical capacity)
    if num_vehicles is None:
        total_demand = customers['demand'].sum()
        estimated_capacity = 200  # Typical capacity for Homberger instances
        num_vehicles = max(1, int(np.ceil(total_demand / estimated_capacity)))
    
    routes_data = []
    
    # Shuffle customers for realistic random assignment
    customers_shuffled = customers.sample(frac=1).reset_index(drop=True)
    
    # Calculate distances (simplified Euclidean)
    def calc_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    customers_per_vehicle = len(customers) // num_vehicles
    extra_customers = len(customers) % num_vehicles
    
    start_idx = 0
    for vehicle_id in range(num_vehicles):
        # Determine how many customers this vehicle serves
        end_idx = start_idx + customers_per_vehicle + (1 if vehicle_id < extra_customers else 0)
        vehicle_customers = customers_shuffled.iloc[start_idx:end_idx]
        
        if len(vehicle_customers) == 0:
            continue
        
        # Start at depot
        current_time = 0
        current_x, current_y = depot['x'], depot['y']
        
        # Add depot as first stop
        routes_data.append({
            'vehicle_id': vehicle_id,
            'customer_id': 0,
            'sequence': 0,
            'arrival_time': 0
        })
        
        # Sort customers by a combination of distance and time windows for realism
        vehicle_customers = vehicle_customers.copy()
        vehicle_customers['distance_from_depot'] = vehicle_customers.apply(
            lambda row: calc_distance(depot['x'], depot['y'], row['x'], row['y']), axis=1
        )
        
        # Sort by ready_time first, then by distance (simple heuristic)
        vehicle_customers = vehicle_customers.sort_values(['ready_time', 'distance_from_depot'])
        
        sequence = 1
        for _, customer in vehicle_customers.iterrows():
            # Calculate travel time (assuming speed of 1 unit/time)
            travel_distance = calc_distance(current_x, current_y, customer['x'], customer['y'])
            travel_time = travel_distance  # Simplified: distance = time
            
            # Arrival time is current time + travel time
            arrival_time = current_time + travel_time
            
            # If arrived before ready time, wait
            arrival_time = max(arrival_time, customer['ready_time'])
            
            # Add service time (random between 5-15 minutes)
            service_time = random.uniform(5, 15)
            
            routes_data.append({
                'vehicle_id': vehicle_id,
                'customer_id': int(customer['customer_id']),
                'sequence': sequence,
                'arrival_time': round(arrival_time, 2)
            })
            
            # Update current position and time
            current_x, current_y = customer['x'], customer['y']
            current_time = arrival_time + service_time
            sequence += 1
        
        # Return to depot
        travel_distance = calc_distance(current_x, current_y, depot['x'], depot['y'])
        travel_time = travel_distance
        return_time = current_time + travel_time
        
        routes_data.append({
            'vehicle_id': vehicle_id,
            'customer_id': 0,
            'sequence': sequence,
            'arrival_time': round(return_time, 2)
        })
        
        start_idx = end_idx
    
    # Convert to polars DataFrame
    routes_df = pl.DataFrame(routes_data)
    return routes_df

def plot_homberger_instance(dataset_path, routes_df: Optional[pl.DataFrame] = None, save_path=None, show_plot=True):
    """
    Plot a Homberger VRPTW instance with timewindows shown as tickers and optional routes.
    
    Args:
        dataset_path (str): Path to the Homberger dataset file
        routes_df (pl.DataFrame, optional): Routes dataframe with columns: vehicle_id, customer_id, sequence, arrival_time
        save_path (str, optional): Path to save the plot. If None, uses dataset filename
        show_plot (bool): Whether to display the plot
    
    Returns:
        dict: Summary statistics of the dataset
    """
    # Load the data
    loader = HombergerDataLoader(dataset_path)
    df = loader.load()
    
    # Convert to pandas for easier plotting
    df_pandas = df.to_pandas()
    
    # Separate depot (customer_id = 0) from customers
    depot = df_pandas[df_pandas['customer_id'] == 0].iloc[0]
    customers = df_pandas[df_pandas['customer_id'] != 0]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot customers
    scatter = ax.scatter(customers['x'], customers['y'], 
                        s=customers['demand']*3,  # Size based on demand
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5,
                        label='Customers')
    
    # Plot depot distinctly
    ax.scatter(depot['x'], depot['y'], 
              c='red', 
              s=200, 
              marker='s',  # Square marker for depot
              edgecolors='black',
              linewidths=2,
              label='Depot',
              zorder=5)
    
    # Plot routes if provided
    if routes_df is not None:
        routes_pandas = routes_df.to_pandas()
        # Get unique vehicles
        vehicles = routes_pandas['vehicle_id'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(vehicles)))
        
        for i, vehicle_id in enumerate(vehicles):
            vehicle_route = routes_pandas[routes_pandas['vehicle_id'] == vehicle_id].sort_values('sequence')
            
            # Get coordinates for this route
            route_coords = []
            for _, route_stop in vehicle_route.iterrows():
                customer_data = df_pandas[df_pandas['customer_id'] == route_stop['customer_id']]
                if len(customer_data) > 0:
                    route_coords.append((customer_data.iloc[0]['x'], customer_data.iloc[0]['y']))
            
            if len(route_coords) > 1:
                # Plot route lines
                xs, ys = zip(*route_coords)
                ax.plot(xs, ys, color=colors[i], linewidth=2, alpha=0.7, 
                       label=f'Vehicle {vehicle_id}', zorder=3)
                
                # Add arrows to show direction
                for j in range(len(xs)-1):
                    dx = xs[j+1] - xs[j]
                    dy = ys[j+1] - ys[j]
                    ax.arrow(xs[j], ys[j], dx*0.7, dy*0.7, 
                           head_width=2, head_length=2, fc=colors[i], ec=colors[i], 
                           alpha=0.6, zorder=4)

    # Add timewindow annotations as tickers
    for idx, row in customers.iterrows():
        # Create timewindow ticker text
        timewindow_text = f"[{row['ready_time']:.0f}-{row['due_time']:.0f}]"
        ax.annotate(timewindow_text, 
                    (row['x'], row['y']), 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    fontsize=6,
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'),
                    zorder=10)
    
    # Add annotations for some key information
    ax.annotate(f'Depot\n({depot["x"]:.0f}, {depot["y"]:.0f})', 
                (depot['x'], depot['y']), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=8)
    
    # Extract instance name from path
    instance_name = os.path.basename(dataset_path).replace('.TXT', '')
    
    # Add title and labels
    ax.set_title(f'Homberger VRPTW Instance - {instance_name}\nCustomer Locations with Time Windows', fontsize=14)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with instance information
    route_info = ""
    if routes_df is not None:
        num_vehicles = len(routes_df.to_pandas()['vehicle_id'].unique())
        route_info = f"\n    • Routes: {num_vehicles} vehicles shown"
    
    info_text = f"""Instance Info:
    • Depot: ({depot['x']:.0f}, {depot['y']:.0f})
    • Customers: {len(customers)}
    • Point size ∝ Demand
    • Time windows shown as [ready-due]
    • Due times: {customers['due_time'].min():.0f} - {customers['due_time'].max():.0f}{route_info}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f'{instance_name}_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    # Create summary statistics
    summary = {
        'total_locations': len(df_pandas),
        'depot_location': (depot['x'], depot['y']),
        'num_customers': len(customers),
        'ready_time_range': (customers['ready_time'].min(), customers['ready_time'].max()),
        'due_time_range': (customers['due_time'].min(), customers['due_time'].max()),
        'demand_range': (customers['demand'].min(), customers['demand'].max())
    }
    
    return summary

# Example usage
if __name__ == "__main__":
    dataset_path = '../../data/homberger_200_customer_instances/C1_2_1.TXT'
    summary = plot_homberger_instance(dataset_path)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total locations: {summary['total_locations']}")
    print(f"Depot location: {summary['depot_location']}")
    print(f"Customer ready times: {summary['ready_time_range'][0]} - {summary['ready_time_range'][1]}")
    print(f"Customer due times: {summary['due_time_range'][0]} - {summary['due_time_range'][1]}")
    print(f"Customer demands: {summary['demand_range'][0]} - {summary['demand_range'][1]}")