"""
Gradio Interface for VRPTW Solver

A simple web interface that allows users to launch datasets and run the solver
with visualization of the results.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path so we can import src modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import gradio as gr
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import glob

# Import solver components
from src.solver.solver import VrptwSolver, create_sample_problem
from src.solver.domain import VrptwSolution, Vehicle, Customer, Location, DriverProfile
from src.io.dataloaders import HombergerDataLoader


def get_available_datasets() -> List[str]:
    """Get list of available Homberger dataset files"""
    datasets = ["Sample Dataset"]

    # Get datasets from homberger_200_customer_instances
    data_dir = Path("data/homberger_200_customer_instances")
    if data_dir.exists():
        pattern = str(data_dir / "*.TXT")
        for file_path in sorted(glob.glob(pattern)):
            filename = Path(file_path).stem
            datasets.append(f"homberger/{filename}")

    # Get datasets from scenario_2
    scenario2_dir = Path("data/scenario_2")
    if scenario2_dir.exists():
        pattern = str(scenario2_dir / "*.TXT")
        for file_path in sorted(glob.glob(pattern)):
            filename = Path(file_path).stem
            datasets.append(f"scenario_2/{filename}")

    return datasets


def load_homberger_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load a Homberger dataset and convert to our format"""
    if dataset_name == "Sample Dataset":
        return create_sample_dataset()

    try:
        # Parse dataset name to get directory and filename
        if dataset_name.startswith("homberger/"):
            filename = dataset_name.replace("homberger/", "")
            file_path = f"data/homberger_200_customer_instances/{filename}.TXT"
        elif dataset_name.startswith("scenario_2/"):
            filename = dataset_name.replace("scenario_2/", "")
            file_path = f"data/scenario_2/{filename}.TXT"
        else:
            # Legacy support for old format
            file_path = f"data/homberger_200_customer_instances/{dataset_name}.TXT"

        loader = HombergerDataLoader(file_path)
        df = loader.load()

        # Convert to our format
        depot_row = df.filter(df["customer_id"] == 0)
        customer_rows = df.filter(df["customer_id"] != 0)

        # Create depot
        depot = {
            "x": float(depot_row["x"][0]),
            "y": float(depot_row["y"][0]),
            "name": "Depot"
        }

        # Create vehicles (use capacity from loader)
        vehicles = [
            {"id": i+1, "driver": f"Driver {i+1}", "capacity": 200}
            for i in range(min(25, len(customer_rows)))  # Limit vehicles
        ]

        # Create customers
        customers = []
        for i, row in enumerate(customer_rows.iter_rows()):
            customers.append({
                "id": row[0],  # customer_id
                "x": float(row[1]),  # x
                "y": float(row[2]),  # y
                "demand": int(row[3]),  # demand
                "ready_time": int(row[4]),  # ready_time
                "due_time": int(row[5]),  # due_time
                "service_time": int(row[6]) if len(row) > 6 else 0,  # service_time
                "is_premium": bool(row[7]) if len(row) > 7 else False  # is_premium
            })

            # Limit to reasonable size for demo
            if len(customers) >= 50:
                break

        return {
            "depot": depot,
            "vehicles": vehicles,
            "customers": customers
        }

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return create_sample_dataset()


def create_sample_dataset() -> Dict[str, Any]:
    """Create a sample dataset for demonstration"""
    return {
        "depot": {"x": 50, "y": 50, "name": "Depot"},
        "vehicles": [
            {"id": 1, "driver": "Driver 1", "capacity": 200},
            {"id": 2, "driver": "Driver 2", "capacity": 200},
            {"id": 3, "driver": "Driver 3", "capacity": 200}
        ],
        "customers": [
            {"id": 1, "x": 20, "y": 30, "demand": 25, "ready_time": 0, "due_time": 100, "is_premium": False},
            {"id": 2, "x": 80, "y": 70, "demand": 30, "ready_time": 0, "due_time": 120, "is_premium": False},
            {"id": 3, "x": 40, "y": 80, "demand": 20, "ready_time": 0, "due_time": 150, "is_premium": False},
            {"id": 4, "x": 60, "y": 20, "demand": 35, "ready_time": 0, "due_time": 110, "is_premium": False},
            {"id": 5, "x": 30, "y": 60, "demand": 40, "ready_time": 0, "due_time": 140, "is_premium": False},
            {"id": 6, "x": 70, "y": 40, "demand": 15, "ready_time": 0, "due_time": 130, "is_premium": False},
            {"id": 7, "x": 90, "y": 20, "demand": 45, "ready_time": 0, "due_time": 100, "is_premium": False},
            {"id": 8, "x": 10, "y": 70, "demand": 25, "ready_time": 0, "due_time": 160, "is_premium": False}
        ]
    }


def convert_to_solver_format(dataset: Dict[str, Any]) -> 'VrptwSolution':
    """Convert dataset dictionary to solver domain objects"""
    try:
        # Create depot
        depot_data = dataset['depot']
        depot = Location(id=0, x=depot_data['x'], y=depot_data['y'], name=depot_data.get('name', 'Depot'))

        # Create driver profiles and vehicles
        vehicles = []
        for i, vehicle_data in enumerate(dataset['vehicles']):
            driver = DriverProfile(id=vehicle_data['id'], name=vehicle_data['driver'])
            vehicle = Vehicle(
                id=vehicle_data['id'],
                driver=driver,
                home_location=depot,
                capacity=vehicle_data['capacity']
            )
            vehicles.append(vehicle)

        # Create customers
        customers = []
        for customer_data in dataset['customers']:
            location = Location(
                id=customer_data['id'],
                x=customer_data['x'],
                y=customer_data['y'],
                name=f"Customer {customer_data['id']}"
            )
            customer = Customer(
                id=customer_data['id'],
                location=location,
                demand=customer_data['demand'],
                ready_time=customer_data['ready_time'],
                due_time=customer_data['due_time'],
                service_time=customer_data.get('service_time', 0),
                is_premium=customer_data.get('is_premium', False)
            )
            customers.append(customer)

        return VrptwSolution(depot=depot, customers=customers, vehicles=vehicles)

    except Exception as e:
        print(f"Error converting dataset: {e}")
        return create_sample_problem()


def visualize_dataset(dataset: Dict[str, Any]) -> plt.Figure:
    """Create a matplotlib visualization of just the dataset (no routes)"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot depot
    depot = dataset['depot']
    ax.scatter(depot['x'], depot['y'], c='black', s=200, marker='s', label='Depot')
    ax.annotate('Depot', (depot['x'], depot['y']), xytext=(5, 5), textcoords='offset points')

    # Plot customers
    customers = dataset['customers']
    for customer in customers:
        ax.scatter(customer['x'], customer['y'], c='lightblue', s=100, marker='o')
        ax.annotate(f'C{customer["id"]}\n({customer["demand"]})',
                   (customer['x'], customer['y']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Dataset: {len(customers)} customers, {len(dataset["vehicles"])} vehicles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_solution(solution: 'VrptwSolution') -> plt.Figure:
    """Create a matplotlib visualization of the solution"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Colors for different vehicles
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Plot depot
    ax.scatter(solution.depot.x, solution.depot.y, c='black', s=200, marker='s', label='Depot')
    ax.annotate('Depot', (solution.depot.x, solution.depot.y), xytext=(5, 5), textcoords='offset points')

    # Plot customers
    for customer in solution.customers:
        ax.scatter(customer.location.x, customer.location.y, c='lightblue', s=100, marker='o')
        ax.annotate(f'C{customer.id}\n({customer.demand})',
                   (customer.location.x, customer.location.y),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot routes
    used_vehicles = 0
    for i, vehicle in enumerate(solution.vehicles):
        if len(vehicle.visits) > 0:
            used_vehicles += 1
            color = colors[i % len(colors)]

            # Draw route
            route_x = [solution.depot.x]
            route_y = [solution.depot.y]

            for customer in vehicle.visits:
                route_x.append(customer.location.x)
                route_y.append(customer.location.y)

            # Return to depot
            route_x.append(solution.depot.x)
            route_y.append(solution.depot.y)

            ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.7,
                   label=f'{vehicle.driver.name} ({len(vehicle.visits)} stops)')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'VRPTW Solution - {used_vehicles} vehicles used')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_solver(dataset_json: str, solving_time: int) -> tuple:
    """Run the VRPTW solver and return results"""
    try:
        # Parse dataset
        if dataset_json.strip():
            dataset = json.loads(dataset_json)
        else:
            dataset = create_sample_dataset()

        # Convert to solver format
        problem = convert_to_solver_format(dataset)

        # Create and run solver with default settings
        solver = VrptwSolver(solving_time_seconds=solving_time)
        solution = solver.solve(problem)

        # Generate visualization
        fig = visualize_solution(solution)

        # Generate results summary
        total_distance = solution.get_total_distance()
        active_vehicles = len([v for v in solution.vehicles if len(v.visits) > 0])
        unassigned = len(solution.get_unassigned_customers()) if hasattr(solution, 'get_unassigned_customers') else 0

        results_text = f"""
## Solution Results

**Score:** {solution.score if hasattr(solution, 'score') else 'N/A'}
**Total Distance:** {total_distance:.1f} units
**Active Vehicles:** {active_vehicles}
**Unassigned Customers:** {unassigned}

**Vehicle Routes:**
"""

        for vehicle in solution.vehicles:
            if len(vehicle.visits) > 0:
                customer_ids = [c.id for c in vehicle.visits]
                total_demand = sum(c.demand for c in vehicle.visits)
                route_distance = vehicle.get_total_distance()

                results_text += f"""
- **{vehicle.driver.name}**: {len(vehicle.visits)} stops, Customers: {customer_ids}, Demand: {total_demand}/{vehicle.capacity}, Distance: {route_distance:.1f}
"""

        return fig, results_text

    except Exception as e:
        error_msg = f"Error running solver: {str(e)}"
        print(error_msg)

        # Create empty plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Solver Error")

        return fig, error_msg


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface"""

    with gr.Blocks(title="VRPTW Solver", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# VRPTW Solver")
        gr.Markdown("Load a dataset and run the solver to visualize optimized vehicle routes.")

        with gr.Tabs():
            with gr.Tab("Solver"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Dataset selector
                        dataset_dropdown = gr.Dropdown(
                            choices=get_available_datasets(),
                            value="Sample Dataset",
                            label="Select Dataset",
                            interactive=True
                        )

                        # Solver configuration
                        solving_time = gr.Slider(
                            minimum=5,
                            maximum=120,
                            value=30,
                            step=5,
                            label="Solving Time (seconds)"
                        )

                        # Run button
                        run_button = gr.Button("🚚 Run Solver", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        # Dataset visualization (shows immediately)
                        initial_dataset = create_sample_dataset()
                        initial_viz = visualize_dataset(initial_dataset)
                        dataset_plot = gr.Plot(value=initial_viz, label="Dataset Visualization")

                        # Results visualization (shows after solver runs)
                        solution_plot = gr.Plot(label="Solution Routes", visible=False)

                        # Results text
                        initial_info = f"""
## Dataset: Sample Dataset

**Summary:**
- Customers: {len(initial_dataset['customers'])}
- Vehicles: {len(initial_dataset['vehicles'])}
- Depot: ({initial_dataset['depot']['x']:.1f}, {initial_dataset['depot']['y']:.1f})

Click 'Run Solver' to optimize routes.
"""
                        results_output = gr.Markdown(value=initial_info)

                # Store current dataset in state
                current_dataset = gr.State(value=initial_dataset)

                # Event handlers
                def load_and_visualize_dataset(dataset_name):
                    """Load dataset and return visualization"""
                    dataset = load_homberger_dataset(dataset_name)
                    dataset_viz = visualize_dataset(dataset)

                    info_text = f"""
## Dataset: {dataset_name}

**Summary:**
- Customers: {len(dataset['customers'])}
- Vehicles: {len(dataset['vehicles'])}
- Depot: ({dataset['depot']['x']:.1f}, {dataset['depot']['y']:.1f})

Click 'Run Solver' to optimize routes.
"""

                    return dataset_viz, info_text, gr.update(visible=False), dataset

                dataset_dropdown.change(
                    fn=load_and_visualize_dataset,
                    inputs=[dataset_dropdown],
                    outputs=[dataset_plot, results_output, solution_plot, current_dataset]
                )

                # Update run solver to use current dataset
                def run_solver_and_show(dataset, solving_time):
                    dataset_json_str = json.dumps(dataset, indent=2)
                    solution_viz, results_text = run_solver(dataset_json_str, solving_time)
                    return solution_viz, results_text, gr.update(visible=True)

                run_button.click(
                    fn=run_solver_and_show,
                    inputs=[current_dataset, solving_time],
                    outputs=[solution_plot, results_output, solution_plot]
                )

            with gr.Tab("Edit Dataset"):
                gr.Markdown("## Dataset Editor")
                gr.Markdown("Modify the selected dataset by editing the JSON directly.")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Dataset selector for editor tab
                        edit_dataset_dropdown = gr.Dropdown(
                            choices=get_available_datasets(),
                            value="Sample Dataset",
                            label="Select Dataset to Edit",
                            interactive=True
                        )

                        # Load button to refresh from main tab selection
                        load_current_button = gr.Button("Load Currently Selected Dataset", variant="secondary")

                    with gr.Column(scale=2):
                        pass

                # Dataset JSON editor
                dataset_json_editor = gr.Code(
                    value=json.dumps(create_sample_dataset(), indent=2),
                    language="json",
                    label="Dataset JSON (editable)",
                    lines=25
                )

                # Buttons
                with gr.Row():
                    validate_button = gr.Button("Validate JSON", variant="secondary")
                    apply_button = gr.Button("Apply Changes", variant="primary")

                # Results for editor
                edit_results = gr.Markdown(value="Select a dataset to edit it here.")

                # Event handlers for editor tab
                def load_dataset_for_editing(dataset_name):
                    dataset = load_homberger_dataset(dataset_name)
                    dataset_json_str = json.dumps(dataset, indent=2)
                    return dataset_json_str, f"Loaded {dataset_name} for editing."

                edit_dataset_dropdown.change(
                    fn=load_dataset_for_editing,
                    inputs=[edit_dataset_dropdown],
                    outputs=[dataset_json_editor, edit_results]
                )

                def validate_json(json_str):
                    try:
                        dataset = json.loads(json_str)
                        # Basic validation
                        required_keys = ['depot', 'vehicles', 'customers']
                        for key in required_keys:
                            if key not in dataset:
                                return f"❌ Missing required key: {key}"

                        return f"✅ Valid JSON with {len(dataset['customers'])} customers and {len(dataset['vehicles'])} vehicles"
                    except json.JSONDecodeError as e:
                        return f"❌ Invalid JSON: {str(e)}"

                validate_button.click(
                    fn=validate_json,
                    inputs=[dataset_json_editor],
                    outputs=[edit_results]
                )

            with gr.Tab("Analysis"):
                gr.Markdown("## Route Analysis (Coming Soon)")
                gr.Markdown("Compare different solutions and analyze route quality metrics.")

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=True,  # Create shareable link
        debug=True,  # Enable debug mode
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860  # Standard Gradio port
    )