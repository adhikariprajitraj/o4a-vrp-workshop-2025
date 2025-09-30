# O4A - VRP Workshop 2025 
**October 2025**

A hands-on workshop for solving Vehicle Routing Problems (VRP) with Time Windows using Timefold Solver. Participants will learn to customize optimization constraints and build their own routing solutions.

## 🚀 Quick Start

### Prerequisites
- Java >= 17
- Python 3.10.12
- pip (Python package manager)

> [!NOTE]
> Timefold requires the JVM to run, so you'll need to install that first

**macOS:**
```bash
brew install openjdk
```

After installation, symlink it for macOS to recognize it:
```bash
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

**Windows:**
1. Download OpenJDK from [Adoptium](https://adoptium.net/) or [Oracle](https://www.oracle.com/java/technologies/downloads/)
2. Run the installer and follow the installation wizard
3. The installer will automatically set up the `JAVA_HOME` environment variable

Verify installation:
```bash
java -version
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd o4a-vrp-workshop-2025
   ```

2. **Create a virtual environment** (recended):
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Run the web interface**:
   ```bash
   gradio gradio_interface.py
   ```
   Using the `gradio` command allos the interface to listen to changes in the code and a refresh is sufficient to load them in
2. **Access the web interface**:
   Open your browser and go to `http://localhost:7860`

3. **Alternative: Run the solver directly**:
   ```bash
   python -m src.solver.solver
   ```

## 🎯 Adding Custom Constraints

The easiest way to customize the routing optimization is by adding your own constraints. Here's how:

### Step 1: Open the Constraints File
Navigate to `src/solver/constraints.py`

### Step 2: Define Your Constraint Function
Copy and modify the example template:

```python
def my_custom_constraint(constraint_factory):
    """Your constraint description here"""
    return (constraint_factory
            .for_each(Vehicle)
            # Your condition here (when to apply penalty)
            .filter(lambda vehicle: len(vehicle.visits) > 8)
            .penalize(HardMediumSoftScore.ONE_SOFT,
                     # Your penalty calculation here
                     lambda vehicle: (len(vehicle.visits) - 8) * 50)
            .as_constraint("My constraint name"))
```

### Step 3: Add to Active Constraints List
Find the `ACTIVE_CONSTRAINTS` list at the bottom of the file and add your function:

```python
ACTIVE_CONSTRAINTS = [
    vehicle_capacity_constraint,
    time_window_constraint,
    minimize_vehicles_constraint,

    # Add your constraint here:
    my_custom_constraint,
]
```

### Step 4: Run the Application
Your constraint will automatically be loaded and used by the solver!


## 🔧 Constraint Levels

Use different penalty levels to prioritize your constraints:

- **`HardMediumSoftScore.ONE_HARD`**: Must be satisfied (violation = infeasible solution)
- **`HardMediumSoftScore.ONE_MEDIUM`**: High priority (like minimizing vehicles)
- **`HardMediumSoftScore.ONE_SOFT`**: Lower priority optimization goals

## 📊 Available Data

### Vehicle Properties
- `vehicle.visits`: List of assigned customers
- `vehicle.capacity`: Maximum capacity
- `vehicle.home_location`: Depot location
- `vehicle.get_total_demand()`: Total demand of assigned customers
- `vehicle.get_total_distance()`: Total travel distance

### Customer Properties
- `customer.location`: Location with x, y coordinates
- `customer.demand`: Resource demand (weight, volume, etc.)
- `customer.ready_time`: Earliest service time
- `customer.due_time`: Latest service time
- `customer.service_time`: Time needed to serve customer

## 🛠️ Project Structure

```
o4a-vrp-workshop-2025/
├── src/
│   ├── solver/
│   │   ├── constraints.py    # Add your constraints here!
│   │   ├── domain.py         # Data models
│   │   └── solver.py         # Main solver logic
│   └── io/
│       └── dataloaders.py    # Data loading utilities
├── data/                     # Dataset files
├── gradio_interface.py       # Web interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎓 Workshop Tips

1. **Start Simple**: Begin by modifying the example constraint
2. **Test Incrementally**: Add one constraint at a time
3. **Use Helper Functions**: Break complex logic into separate functions
4. **Check the Logs**: The solver shows which constraints are active
5. **Experiment**: Try different penalty values and see how they affect routes

## 📈 Testing Your Constraints

1. Run the application: `gradio gradio_interface.py`
2. Load a dataset in the web interface
3. Click "Solve" to see your constraints in action
4. Check the console output for constraint details
5. Visualize the results on the map

## 🚨 Troubleshooting

### Timefold Installation Issues
If you encounter issues installing Timefold:
```bash
pip install --upgrade pip
pip install timefold --no-cache-dir
```

### Application Won't Start
1. Check Python version: `python --version` (should be 3.8+)
2. Verify all packages are installed: `pip list | grep -E "(timefold|gradio|matplotlib)"`
3. Try running the solver directly: `python -m src.solver.solver`

### Constraint Errors
1. Check syntax in your constraint function
2. Ensure your constraint is added to `ACTIVE_CONSTRAINTS`
3. Check console output for error messages

## 📝 License

This workshop is for educational purposes. Please check individual package licenses for production use.
