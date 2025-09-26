from abc import ABC, abstractmethod
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Load data and return as a Polars DataFrame."""
        pass


class HombergerDataLoader(DataLoader):
    """Data loader for Gehring & Homberger VRPTW instances."""
    
    def __init__(self, file_path: str):
        """
        Initialize the Homberger data loader.
        
        Args:
            file_path: Path to the Homberger instance file
        """
        self.file_path = Path(file_path)
        
    def load(self) -> pl.DataFrame:
        """
        Load Gehring & Homberger instance data.
        
        Returns:
            Polars DataFrame with columns: customer_id, x, y, demand, ready_time, due_time, service_time
        """
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        parsed_data = self._parse_homberger_format(content)
        
        return pl.DataFrame(parsed_data['customers'])
    
    def _parse_homberger_format(self, content: str) -> Dict[str, Any]:
        """
        Parse Homberger format file content.
        
        Args:
            content: Raw file content
            
        Returns:
            Dictionary with parsed data
        """
        lines = content.strip().split('\n')
        
        # Parse instance name (line 1)
        instance_name = lines[0].strip()
        
        # Parse vehicle information (line 5)
        vehicle_num, capacity = map(int, lines[4].split())
        
        # Parse header information (lines 7-8)
        customer_section_header = lines[6].strip()  # "CUSTOMER" 
        column_headers = lines[7].strip()  # Column header line
        
        # Parse customer data (starting from line 10)
        customers = []
        for line in lines[9:]:  # Customer data starts at line 10 (index 9)
            if line.strip():
                parts = line.split()
                if len(parts) >= 7:  # Ensure we have all required fields
                    customer = {
                        'customer_id': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'demand': int(parts[3]),
                        'ready_time': int(parts[4]),
                        'due_time': int(parts[5]),
                        'service_time': int(parts[6])
                    }
                    customers.append(customer)
        
        return {
            'instance_name': instance_name,
            'vehicle_num': vehicle_num,
            'capacity': capacity,
            'customer_section_header': customer_section_header,
            'column_headers': column_headers,
            'customers': customers
        }