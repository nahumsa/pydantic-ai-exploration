from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_df(
    n_rows: int = 1000, start_date: datetime = datetime(2022, 1, 1)
) -> pd.DataFrame:
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    makes = [
        "Toyota",
        "Honda",
        "Ford",
        "Chevrolet",
        "Nissan",
        "BMW",
        "Mercedes",
        "Audi",
        "Hyundai",
        "Kia",
    ]
    models = ["Sedan", "SUV", "Truck", "Hatchback", "Coupe", "Van"]
    colors = ["Red", "Blue", "Black", "White", "Silver", "Gray", "Green"]

    data = {
        "Date": dates,
        "Make": np.random.choice(makes, n_rows),
        "Model": np.random.choice(models, n_rows),
        "Color": np.random.choice(colors, n_rows),
        "Year": np.random.randint(2015, 2023, n_rows),
        "Price": np.random.uniform(20000, 80000, n_rows).round(2),
        "Mileage": np.random.uniform(0, 100000, n_rows).round(0),
        "EngineSize": np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
        "FuelEfficiency": np.random.uniform(20, 40, n_rows).round(1),
        "SalesPerson": np.random.choice(
            ["Alice", "Bob", "Charlie", "David", "Eva"], n_rows
        ),
    }

    return pd.DataFrame(data).sort_values("Date")
