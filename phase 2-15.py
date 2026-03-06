import simpy
import random
import math
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

# --- Global Configuration & Constants ---
MAP_SIZE = 20
BASE_FARE = 3.0
FARE_PER_MILE = 2.0
COST_PER_MILE = 0.20
PATIENCE_RATE = 5.0

# 1) Data Loading and Cleaning
df_drivers = pd.read_excel("drivers (1).xlsx")
df_riders = pd.read_excel("riders (1).xlsx")

# Strip whitespace from column names
df_drivers.columns = [c.strip() for c in df_drivers.columns]
df_riders.columns = [c.strip() for c in df_riders.columns]

# Convert time columns to numeric format
for c in ["arrival_time", "offline_time"]:
    df_drivers[c] = pd.to_numeric(df_drivers[c], errors="coerce")
for c in ["request_time", "pickup_time", "dropoff_time"]:
    df_riders[c] = pd.to_numeric(df_riders[c], errors="coerce")

# Define abandonment criteria based on status or missing times
if "status" in df_riders.columns:
    status_lower = df_riders["status"].astype(str).str.lower()
else:
    status_lower = pd.Series([""] * len(df_riders), index=df_riders.index)

df_riders["is_abandoned"] = (
    (df_riders["pickup_time"] == -1) | 
    (df_riders["dropoff_time"] == -1) | 
    (status_lower == "abandoned")
)

# Calculate driver online duration in hours
df_drivers["online_duration"] = df_drivers["offline_time"] - df_drivers["arrival_time"]
df_drivers_clean = df_drivers.dropna(subset=["arrival_time", "offline_time", "online_duration"]).copy()
df_drivers_clean = df_drivers_clean[df_drivers_clean["online_duration"] > 0].reset_index(drop=True)

# Calibrate arrival rates (N / total time window)
def rate_by_window(time_series: pd.Series) -> float:
    t = time_series.dropna().values
    if len(t) < 2: return np.nan
    window = t.max() - t.min()
    return (len(t) / window) if window > 0 else np.nan

RIDER_ARRIVAL = rate_by_window(df_riders["request_time"])
DRIVER_ARRIVAL = rate_by_window(df_drivers_clean["arrival_time"])

# 2) Location Parsing
def parse_location(val):
    if pd.isna(val): return (np.nan, np.nan)
    s = str(val).strip()
    try:
        # Attempt to parse coordinates from strings like "(x, y)" or "[x, y]"
        parts = s.replace("(","").replace(")","").replace("[","").replace("]","").split(",")
        if len(parts) == 2: return (float(parts[0]), float(parts[1]))
    except: pass
    return (np.nan, np.nan)

def add_xy_columns(df, location_col, prefix):
    xy = df[location_col].apply(parse_location)
    df[f"{prefix}_x"] = xy.apply(lambda t: t[0])
    df[f"{prefix}_y"] = xy.apply(lambda t: t[1])
    return df

# 3) Speed Diagnostics and Time Ratio Pool
df_riders = add_xy_columns(df_riders, "pickup_location", "pickup")
df_riders = add_xy_columns(df_riders, "dropoff_location", "dropoff")
df_trips = df_riders[(~df_riders["is_abandoned"]) & df_riders["pickup_time"].notna()].copy()

df_trips["trip_distance"] = np.sqrt((df_trips["pickup_x"] - df_trips["dropoff_x"])**2 + 
                                    (df_trips["pickup_y"] - df_trips["dropoff_y"])**2)
df_trips["trip_time"] = df_trips["dropoff_time"] - df_trips["pickup_time"]
df_trips["actual_speed"] = df_trips["trip_distance"] / df_trips["trip_time"]
mean_speed = df_trips["actual_speed"].mean()
SPEED = float(mean_speed)

# Create empirical time-ratio pool to model travel time variability
df_trips["time_ratio"] = df_trips["trip_time"] / (df_trips["trip_distance"] / SPEED)
lo, hi = df_trips["time_ratio"].quantile([0.01, 0.99])
ratio_pool = df_trips[(df_trips["time_ratio"] >= lo) & (df_trips["time_ratio"] <= hi)]["time_ratio"].values

# 4) Simulation Model Classes

def get_random_location():
    """Generates a random coordinate within the 20x20 square."""
    return (random.uniform(0, MAP_SIZE), random.uniform(0, MAP_SIZE))

def calc_distance(loc1, loc2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

def calc_trip_time(dist):
    """Calculates trip time using mean speed and empirical variability."""
    if dist <= 0: return 0.0
    base_time = dist / SPEED
    ratio = float(np.random.choice(ratio_pool)) if len(ratio_pool) > 0 else 1.0
    return base_time * ratio

class Dispatcher:
    """Manages the matching logic between idle drivers and waiting riders."""
    def __init__(self, env):
        self.env = env
        self.idle_drivers = []
        self.waiting_riders = []
        self.logs = {"completed_trips": 0, "abandoned_trips": 0, "rider_wait_times": [], "driver_stats": []}

    def match(self):
        """Matches the closest available driver to the first waiting rider."""
        if not self.idle_drivers or not self.waiting_riders: return
        for rider in list(self.waiting_riders):
            if not self.idle_drivers: break
            closest_driver = min(self.idle_drivers, key=lambda d: calc_distance(d.location, rider.origin))
            self.idle_drivers.remove(closest_driver)
            self.waiting_riders.remove(rider)
            rider.matched_event.succeed(value=closest_driver)
            if closest_driver.assigned_event: closest_driver.assigned_event.succeed(value=rider)

class Driver:
    """Represents a taxi driver and their lifecycle."""
    def __init__(self, env, dispatcher, driver_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = driver_id
        self.location = get_random_location()
        self.start_time = env.now
        # Sample shift duration from empirical driver data
        self.planned_shift = float(np.random.choice(df_drivers_clean["online_duration"].values))
        self.planned_offline_time = self.start_time + self.planned_shift
        self.total_revenue = 0.0
        self.total_distance_driven = 0.0
        self.trips_completed = 0
        env.process(self.run())

    def run(self):
        """Main driver loop: wait for assignment or end of shift."""
        while self.env.now < self.planned_offline_time:
            self.assigned_event = self.env.event()
            self.dispatcher.idle_drivers.append(self)
            self.dispatcher.match()
            
            time_left = self.planned_offline_time - self.env.now
            results = yield self.assigned_event | self.env.timeout(max(0, time_left))
            
            if self.assigned_event in results:
                rider = self.assigned_event.value
                # Leg 1: Drive to pick up the rider (deadhead)
                dist_to_pickup = calc_distance(self.location, rider.origin)
                self.total_distance_driven += dist_to_pickup
                yield self.env.timeout(calc_trip_time(dist_to_pickup))
                
                # Record wait time for rider satisfaction
                self.dispatcher.logs["rider_wait_times"].append(self.env.now - rider.request_time)
                
                # Leg 2: Drive rider to destination
                dist_to_dropoff = calc_distance(rider.origin, rider.destination)
                self.total_distance_driven += dist_to_dropoff
                yield self.env.timeout(calc_trip_time(dist_to_dropoff))
                
                # Calculate financial earnings
                self.total_revenue += BASE_FARE + (FARE_PER_MILE * dist_to_dropoff)
                self.location = rider.destination
                self.trips_completed += 1
                self.dispatcher.logs["completed_trips"] += 1
            else:
                # Shift ended while driver was idle
                if self in self.dispatcher.idle_drivers: self.dispatcher.idle_drivers.remove(self)
                break
        
        # Log final stats once driver goes offline
        actual_shift = self.env.now - self.start_time
        net_profit = self.total_revenue - (self.total_distance_driven * COST_PER_MILE)
        self.dispatcher.logs["driver_stats"].append({
            "id": self.id, "shift_length": actual_shift, "net_profit": net_profit,
            "hourly_wage": net_profit / actual_shift if actual_shift > 0 else 0,
            "delayed_rest": max(0, self.env.now - self.planned_offline_time)
        })

class Rider:
    """Represents a customer requesting a ride."""
    def __init__(self, env, dispatcher, rider_id):
        self.env, self.dispatcher, self.id = env, dispatcher, rider_id
        self.origin, self.destination = get_random_location(), get_random_location()
        self.request_time = env.now
        self.matched_event = env.event()
        env.process(self.run())

    def run(self):
        """Rider loop: wait for matching within patience window."""
        self.dispatcher.waiting_riders.append(self)
        self.dispatcher.match()
        patience = random.expovariate(PATIENCE_RATE)
        results = yield self.matched_event | self.env.timeout(patience)
        if self.matched_event not in results:
            if self in self.dispatcher.waiting_riders: self.dispatcher.waiting_riders.remove(self)
            self.dispatcher.logs["abandoned_trips"] += 1

# --- Execution Logic ---

def driver_generator(env, dispatcher):
    """Generates new drivers based on calibrated arrival rate."""
    d_id = 0
    while True:
        yield env.timeout(random.expovariate(DRIVER_ARRIVAL))
        d_id += 1
        Driver(env, dispatcher, d_id)

def rider_generator(env, dispatcher):
    """Generates new riders based on calibrated arrival rate."""
    r_id = 0
    while True:
        yield env.timeout(random.expovariate(RIDER_ARRIVAL))
        r_id += 1
        Rider(env, dispatcher, r_id)

def run_one_replication(sim_time=720):
    """Executes a single simulation run for the specified time (hours)."""
    env = simpy.Environment()
    dispatcher = Dispatcher(env)
    env.process(driver_generator(env, dispatcher))
    env.process(rider_generator(env, dispatcher))
    env.run(until=sim_time)
    
    # Summary calculations would go here
    return dispatcher.logs

# Run 15 independent replications (30 days each)
all_results = []
for rep in range(15):
    random.seed(1000 + rep)
    np.random.seed(2000 + rep)
    all_results.append(run_one_replication(sim_time=720))

print("\n" + "="*40)
print(" PHASE 2 CORRECTED RESULTS (30 Days) ")
print("="*40)