"""
Download famous CSV datasets for testing the chart generator.
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path

def download_file(url: str, filepath: str):
    """Download a file from URL."""
    print(f"Downloading {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    print(f"✅ Downloaded {filepath}")

def create_sample_datasets():
    """Create sample datasets for testing."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # 1. Titanic Dataset (famous for ML)
    print("\n1. Creating Titanic dataset...")
    try:
        titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        titanic_path = datasets_dir / "titanic.csv"
        download_file(titanic_url, str(titanic_path))
        
        # Verify
        df = pd.read_csv(titanic_path)
        print(f"   ✅ Titanic: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   ❌ Error downloading Titanic: {e}")
        # Create a sample instead
        create_sample_titanic(datasets_dir / "titanic.csv")
    
    # 2. Iris Dataset (famous for classification)
    print("\n2. Creating Iris dataset...")
    try:
        iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        iris_path = datasets_dir / "iris.csv"
        download_file(iris_url, str(iris_path))
        
        df = pd.read_csv(iris_path)
        print(f"   ✅ Iris: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   ❌ Error downloading Iris: {e}")
        create_sample_iris(datasets_dir / "iris.csv")
    
    # 3. Sales Sample Dataset (for business analytics)
    print("\n3. Creating Sales dataset...")
    create_sample_sales(datasets_dir / "sales_data.csv")
    
    # 4. Customer Dataset
    print("\n4. Creating Customer dataset...")
    create_sample_customers(datasets_dir / "customer_data.csv")
    
    print("\n✅ All datasets created!")

def create_sample_titanic(filepath: Path):
    """Create a sample Titanic dataset."""
    import random
    random.seed(42)
    
    data = {
        'PassengerId': range(1, 892),
        'Survived': [random.choice([0, 1]) for _ in range(891)],
        'Pclass': [random.choice([1, 2, 3]) for _ in range(891)],
        'Name': [f"Person {i}" for i in range(891)],
        'Sex': [random.choice(['male', 'female']) for _ in range(891)],
        'Age': [random.randint(1, 80) for _ in range(891)],
        'SibSp': [random.randint(0, 5) for _ in range(891)],
        'Parch': [random.randint(0, 3) for _ in range(891)],
        'Ticket': [f"Ticket{i}" for i in range(891)],
        'Fare': [round(random.uniform(7, 500), 2) for _ in range(891)],
        'Cabin': [f"C{random.randint(1, 100)}" if random.random() > 0.5 else None for _ in range(891)],
        'Embarked': [random.choice(['S', 'C', 'Q']) for _ in range(891)]
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"   ✅ Created sample Titanic: {len(df)} rows")

def create_sample_iris(filepath: Path):
    """Create a sample Iris dataset."""
    import random
    random.seed(42)
    
    species = ['setosa', 'versicolor', 'virginica']
    data = {
        'sepal_length': [round(random.uniform(4, 8), 1) for _ in range(150)],
        'sepal_width': [round(random.uniform(2, 4.5), 1) for _ in range(150)],
        'petal_length': [round(random.uniform(1, 7), 1) for _ in range(150)],
        'petal_width': [round(random.uniform(0.1, 2.5), 1) for _ in range(150)],
        'species': [random.choice(species) for _ in range(150)]
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"   ✅ Created sample Iris: {len(df)} rows")

def create_sample_sales(filepath: Path):
    """Create a sample sales dataset."""
    import random
    from datetime import datetime, timedelta
    random.seed(42)
    
    products = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys']
    regions = ['North', 'South', 'East', 'West']
    
    start_date = datetime(2023, 1, 1)
    data = []
    for i in range(500):
        date = start_date + timedelta(days=i % 365)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'product_category': random.choice(products),
            'region': random.choice(regions),
            'sales_amount': round(random.uniform(100, 5000), 2),
            'units_sold': random.randint(1, 100),
            'customer_id': f"CUST{random.randint(1000, 9999)}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"   ✅ Created Sales: {len(df)} rows")

def create_sample_customers(filepath: Path):
    """Create a sample customer dataset."""
    import random
    random.seed(42)
    
    genders = ['Male', 'Female', 'Other']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    data = []
    for i in range(300):
        data.append({
            'customer_id': f"CUST{1000 + i}",
            'age': random.randint(18, 80),
            'gender': random.choice(genders),
            'location': random.choice(locations),
            'total_purchases': random.randint(1, 50),
            'average_order_value': round(random.uniform(20, 500), 2),
            'last_purchase_date': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"   ✅ Created Customers: {len(df)} rows")

if __name__ == "__main__":
    print("="*70)
    print("Downloading/Creating Sample Datasets")
    print("="*70)
    create_sample_datasets()
    print("\n" + "="*70)
    print("Next step: Create metadata.json file")
    print("="*70)
