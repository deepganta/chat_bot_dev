"""
Tables
------
plants(id PK, name, city, state, country, opened_year)
dealers(id PK, name, city, state, region)
vehicles(vin PK, model, model_year, trim, segment, msrp, plant_id FK)
sales(id PK, vin FK, dealer_id FK, sale_date, sale_price, customer_type)

Usage
-----
$ python -m pipelines.etl.generate_ford_db --rows 200 --db data/ford.db --seed 42
"""

import argparse
import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

from faker import Faker

# ----------------------------
# E = EXTRACT / SYNTHESIZE
# ----------------------------

FORD_MODELS = [
    # cars & SUVs across eras to keep variety
    ("F-150", "Truck", ["XL", "XLT", "Lariat", "Platinum", "Raptor"]),
    ("Mustang", "Sports", ["EcoBoost", "GT", "Dark Horse"]),
    ("Explorer", "SUV", ["Base", "XLT", "Limited", "ST"]),
    ("Escape", "SUV", ["Active", "ST-Line", "Platinum"]),
    ("Bronco", "Off-Road", ["Base", "Big Bend", "Black Diamond", "Wildtrak"]),
    ("Maverick", "Truck", ["XL", "XLT", "Lariat"]),
    ("Edge", "SUV", ["SE", "SEL", "Titanium", "ST"]),
    ("Expedition", "SUV", ["XL", "XLT", "Limited", "King Ranch", "Platinum"]),
]

US_REGIONS = ["Northeast", "Midwest", "South", "West"]

COUNTRIES = ["USA", "Canada", "Mexico"]
US_STATES = [
    "MI", "OH", "KY", "MO", "IL", "TX", "CA", "NY", "PA", "FL", "AZ", "CO", "GA", "NC", "TN"
]


def _ensure_dirs(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _drop_and_create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS vehicles;
        DROP TABLE IF EXISTS dealers;
        DROP TABLE IF EXISTS plants;

        CREATE TABLE plants(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            country TEXT NOT NULL,
            opened_year INTEGER NOT NULL
        );

        CREATE TABLE dealers(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            region TEXT NOT NULL
        );

        CREATE TABLE vehicles(
            vin TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            model_year INTEGER NOT NULL,
            trim TEXT NOT NULL,
            segment TEXT NOT NULL,
            msrp REAL NOT NULL,
            plant_id INTEGER NOT NULL,
            FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE RESTRICT
        );

        CREATE TABLE sales(
            id INTEGER PRIMARY KEY,
            vin TEXT NOT NULL,
            dealer_id INTEGER NOT NULL,
            sale_date DATE NOT NULL,
            sale_price REAL NOT NULL,
            customer_type TEXT NOT NULL CHECK(customer_type IN ('Retail','Fleet')),
            FOREIGN KEY (vin) REFERENCES vehicles(vin) ON DELETE CASCADE,
            FOREIGN KEY (dealer_id) REFERENCES dealers(id) ON DELETE RESTRICT
        );
        """
    )
    conn.commit()


def _rand_open_year(rng: random.Random) -> int:
    # Ford plants opened across decades
    return rng.randint(1965, date.today().year - 1)


def _vin(rng: random.Random) -> str:
    # simple pseudo VIN: 17 chars (NOT real VIN spec)
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(17))


def _price_for(model: str, segment: str, trim: str, rng: random.Random) -> float:
    base = {
        "Truck": 33000,
        "SUV": 28000,
        "Off-Road": 36000,
        "Sports": 32000,
    }.get(segment, 30000)
    # model & trim uplift
    model_uplift = {
        "F-150": 8000,
        "Mustang": 9000,
        "Explorer": 5000,
        "Escape": 2000,
        "Bronco": 7000,
        "Maverick": 3000,
        "Edge": 3500,
        "Expedition": 12000,
    }.get(model, 0)
    trim_uplift = {
        "XL": 0, "XLT": 2500, "Lariat": 6500, "Platinum": 12000, "Raptor": 20000,
        "EcoBoost": 0, "GT": 8000, "Dark Horse": 20000,
        "Base": 0, "Limited": 7000, "ST": 9000,
        "Active": 0, "ST-Line": 1800,
        "SE": 0, "SEL": 1800, "Titanium": 4500, "King Ranch": 13000,
        "Big Bend": 1500, "Black Diamond": 4000, "Wildtrak": 15000,
    }.get(trim, 0)
    noise = rng.randint(-1500, 2500)
    return float(base + model_uplift + trim_uplift + noise)


def _random_sale_price(msrp: float, rng: random.Random) -> float:
    # Sales typically around MSRP ± up to ~10%
    return round(msrp * rng.uniform(0.9, 1.08), 2)


# ----------------------------
# T = TRANSFORM
# ----------------------------

def synthesize_plants(n: int, fake: Faker, rng: random.Random) -> List[Tuple]:
    rows = []
    for i in range(1, n + 1):
        city = fake.city()
        state = rng.choice(US_STATES)
        country = "USA"
        name = f"Ford {city} Assembly Plant"
        rows.append((i, name, city, state, country, _rand_open_year(rng)))
    return rows


def synthesize_dealers(n: int, fake: Faker, rng: random.Random) -> List[Tuple]:
    rows = []
    for i in range(1, n + 1):
        city = fake.city()
        state = rng.choice(US_STATES)
        region = rng.choice(US_REGIONS)
        name = f"{fake.last_name()} Ford"
        rows.append((i, name, city, state, region))
    return rows


def synthesize_vehicles(n: int, plant_ids: List[int], fake: Faker, rng: random.Random) -> List[Tuple]:
    rows = []
    for _ in range(n):
        model, segment, trims = rng.choice(FORD_MODELS)
        trim = rng.choice(trims)
        model_year = rng.randint(2016, date.today().year)  # recent-ish
        vin = _vin(rng)
        msrp = round(_price_for(model, segment, trim, rng), 2)
        plant_id = rng.choice(plant_ids)
        rows.append((vin, model, model_year, trim, segment, msrp, plant_id))
    return rows


def synthesize_sales(n: int, vins: List[str], dealer_ids: List[int], rng: random.Random) -> List[Tuple]:
    rows = []
    start = date(2018, 1, 1)
    for i in range(1, n + 1):
        vin = rng.choice(vins)
        dealer_id = rng.choice(dealer_ids)
        sale_date = start + timedelta(days=rng.randint(0, (date.today() - start).days))
        customer_type = rng.choice(["Retail", "Fleet"])
        # sale_price determined later during LOAD where msrp is available
        rows.append((i, vin, dealer_id, sale_date.isoformat(), 0.0, customer_type))
    return rows


# ----------------------------
# L = LOAD
# ----------------------------

def load_all(conn: sqlite3.Connection,
             plants: List[Tuple],
             dealers: List[Tuple],
             vehicles: List[Tuple],
             sales: List[Tuple],
             rng: random.Random):
    cur = conn.cursor()

    cur.executemany(
        "INSERT INTO plants(id, name, city, state, country, opened_year) VALUES(?,?,?,?,?,?)",
        plants,
    )
    cur.executemany(
        "INSERT INTO dealers(id, name, city, state, region) VALUES(?,?,?,?,?)",
        dealers,
    )
    cur.executemany(
        "INSERT INTO vehicles(vin, model, model_year, trim, segment, msrp, plant_id) VALUES(?,?,?,?,?,?,?)",
        vehicles,
    )

    # lookup MSRP for each VIN to derive sale_price
    msrp_lookup = {v[0]: v[5] for v in vehicles}
    sales_loaded = []
    for sid, vin, dealer_id, sale_date, _zero, cust in sales:
        msrp = msrp_lookup.get(vin, 30000.0)
        sale_price = _random_sale_price(msrp, rng)
        sales_loaded.append((sid, vin, dealer_id, sale_date, sale_price, cust))

    cur.executemany(
        "INSERT INTO sales(id, vin, dealer_id, sale_date, sale_price, customer_type) VALUES(?,?,?,?,?,?)",
        sales_loaded,
    )
    conn.commit()


def etl(db_path: Path, rows_per_table: int, seed: int):
    rng = random.Random(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    _ensure_dirs(db_path)
    conn = _connect(db_path)
    _drop_and_create_schema(conn)

    # Synthesize (E/T)
    plants = synthesize_plants(rows_per_table, fake, rng)
    dealers = synthesize_dealers(rows_per_table, fake, rng)
    vehicles = synthesize_vehicles(rows_per_table, [p[0] for p in plants], fake, rng)
    sales = synthesize_sales(rows_per_table, [v[0] for v in vehicles], [d[0] for d in dealers], rng)

    # Load (L)
    load_all(conn, plants, dealers, vehicles, sales, rng)

    # Simple sanity counts
    cur = conn.cursor()
    for t in ("plants", "dealers", "vehicles", "sales"):
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:<10}: {cnt} rows")
    conn.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/ford.db", help="SQLite DB path")
    ap.add_argument("--rows", type=int, default=200, help="Approx rows per table")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    etl(Path(args.db), args.rows, args.seed)
