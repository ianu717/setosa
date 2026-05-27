import pickle
import json
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

casualty_df = pd.read_parquet(BASE_DIR / 'data' / 'casualty_cut.parquet')
vehicle_df = pd.read_parquet(BASE_DIR / 'data' / 'vehicle_cut.parquet')
collision_df = pd.read_parquet(BASE_DIR / 'data' / 'collision_cut.parquet')
collision_indexes_df = pd.read_parquet(BASE_DIR / 'data' / 'indexes_cut.parquet')

def save_pickle(path: Path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def get_day_period_from_time(time: str):
    hour = int(time.split(':')[0])
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def generate_random_collision():
    collision_index = collision_indexes_df.sample(1)[0].iloc[0]
    casualties = casualty_df[casualty_df['collision_index'] == collision_index]
    vehicles = vehicle_df[vehicle_df['collision_index'] == collision_index]
    collisions = collision_df[collision_df['collision_index'] == collision_index]

    return casualties, vehicles, collisions