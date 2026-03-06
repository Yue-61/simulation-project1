import simpy
import random
import math
import numpy as np
from scipy.stats import truncnorm

# ==========================================
# 1. Calibrated System Parameters (Phase 2.5)
# ==========================================
MAP_SIZE = 20
SPEED = 20.6052
DRIVER_ARRIVAL = 4.74    # Actual driver arrival rate
RIDER_ARRIVAL = 34.31     # Actual rider arrival rate
PATIENCE_RATE = 5.0      # Rider patience rate (remains constant)

BASE_FARE = 3.0
FARE_PER_MILE = 2.0
COST_PER_MILE = 0.20

# ==========================================
# 2. Phase 3 Intervention Strategy Parameters
# ==========================================
# Intervention A: Off-duty Protection Mechanism
# Stops assigning new rides 30 mins (0.5h) before the planned end of shift
PROTECTION_BUFFER = 0.5  

# Intervention B: Hub Repositioning (Hotspot Guidance)
# Set based on the calibrated mean destination coordinates
CITY_CENTER = (11.23, 13.26) 
# Threshold distance to trigger repositioning from remote areas
REPOSITION_THRESHOLD = 6.0   
# Distance to drive back toward the city center (deadhead)
REPOSITION_DISTANCE = 3.0    

# ==========================================
# 3. Distribution Helper Functions
# ==========================================
def get_truncnorm(mean, std, clip_a=0, clip_b=20):
    """Generates truncated normal coordinates within the 0-20 map range."""
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)

def get_rider_origin():
    """Generates rider pickup locations based on calibrated data."""
    x = get_truncnorm(7.98, 4.93)
    y = get_truncnorm(12.70, 4.97)
    return (x, y)

def get_rider_destination():
    """Generates rider drop-off locations based on calibrated data."""
    x = get_truncnorm(11.23, 4.54)
    y = get_truncnorm(13.26, 4.17)
    return (x, y)

def get_driver_initial():
    """Generates initial driver availability locations."""
    x = get_truncnorm(9.97, 4.36)
    y = get_truncnorm(11.51, 4.34)
    return (x, y)

def calc_distance(loc1, loc2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calc_trip_time(dist):
    """Calculates trip duration using mean speed and empirical variability."""
    if dist <= 0: return 0.0
    base_time = dist / SPEED
    # Based on actual trip time ratio distribution: Uniform(0.7, 1.3)
    ratio = random.uniform(0.7, 1.3)
    return base_time * ratio

# ==========================================
# 4. Dispatcher Logic
# ==========================================
class Dispatcher:
    """Manages matching between available drivers and waiting riders."""
    def __init__(self, env):
        self.env = env
        self.idle_drivers = []
        self.waiting_riders = []
        self.logs = {
            "completed_trips": 0, "abandoned_trips": 0, 
            "rider_wait_times": [], "driver_stats": []
        }

    def match(self):
        """Matches the closest idle driver to the first rider in the queue."""
        if not self.idle_drivers or not self.waiting_riders:
            return 

        for rider in list(self.waiting_riders):
            if not self.idle_drivers:
                break
            
            # Greedy matching for the closest driver
            closest_driver = min(self.idle_drivers, key=lambda d: calc_distance(d.location, rider.origin))
            
            self.idle_drivers.remove(closest_driver)
            self.waiting_riders.remove(rider)
            
            # Trigger events to proceed with the trip
            rider.matched_event.succeed(value=closest_driver)
            closest_driver.assigned_event.succeed(value=rider)

# ==========================================
# 5. Entity Classes: Driver & Rider
# ==========================================
class Driver:
    def __init__(self, env, dispatcher, driver_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = driver_id
        self.location = get_driver_initial()
        
        self.start_time = env.now
        # Real-world online duration distribution (5-8 hours per the problem description)
        self.planned_shift = random.uniform(6, 8) 
        self.planned_offline_time = self.start_time + self.planned_shift
        
        self.total_revenue = 0.0
        self.total_distance_driven = 0.0
        self.trips_completed = 0
        
        self.assigned_event = None
        env.process(self.run())

    def run(self):
        # PHASE 3 INTERVENTION 1: Off-duty Protection Mechanism
        # Only accept new rides if the current time is more than PROTECTION_BUFFER before shift end
        while self.env.now < (self.planned_offline_time - PROTECTION_BUFFER):
            self.assigned_event = self.env.event()
            self.dispatcher.idle_drivers.append(self)
            self.dispatcher.match()
            
            time_until_protection = (self.planned_offline_time - PROTECTION_BUFFER) - self.env.now
            if time_until_protection <= 0:
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
            # Wait for assignment or until protection buffer starts
            results = yield self.assigned_event | self.env.timeout(time_until_protection)
            
            if self.assigned_event in results:
                rider = self.assigned_event.value
                
                # Trip Leg 1: Drive to pickup location (Deadhead)
                dist_to_pickup = calc_distance(self.location, rider.origin)
                self.total_distance_driven += dist_to_pickup
                yield self.env.timeout(calc_trip_time(dist_to_pickup))
                
                # Record rider wait time
                wait_time = self.env.now - rider.request_time
                self.dispatcher.logs["rider_wait_times"].append(wait_time)
                
                # Trip Leg 2: Drive to destination (Service)
                dist_to_dropoff = calc_distance(rider.origin, rider.destination)
                self.total_distance_driven += dist_to_dropoff
                yield self.env.timeout(calc_trip_time(dist_to_dropoff))
                
                # Financial calculations
                trip_revenue = BASE_FARE + (FARE_PER_MILE * dist_to_dropoff)
                self.total_revenue += trip_revenue
                
                self.location = rider.destination
                self.trips_completed += 1
                self.dispatcher.logs["completed_trips"] += 1

                # PHASE 3 INTERVENTION 2: Hub Repositioning
                # Reposition toward city center if drop-off was in a remote area
                dist_to_center = calc_distance(self.location, CITY_CENTER)
                if dist_to_center > REPOSITION_THRESHOLD:
                    move_ratio = REPOSITION_DISTANCE / dist_to_center
                    new_x = self.location[0] + move_ratio * (CITY_CENTER[0] - self.location[0])
                    new_y = self.location[1] + move_ratio * (CITY_CENTER[1] - self.location[1])
                    
                    self.location = (new_x, new_y)
                    self.total_distance_driven += REPOSITION_DISTANCE
                    # Simulate travel time for repositioning
                    yield self.env.timeout(calc_trip_time(REPOSITION_DISTANCE))
            else:
                # Driver became off-duty while idle
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
        # Final settlement for driver KPIs
        actual_offline_time = self.env.now
        actual_shift_length = actual_offline_time - self.start_time
        # Protection buffer significantly reduces this value
        delayed_rest_time = max(0, actual_offline_time - self.planned_offline_time)
        
        total_cost = self.total_distance_driven * COST_PER_MILE
        net_profit = self.total_revenue - total_cost
        
        if actual_shift_length > 0:
            self.dispatcher.logs["driver_stats"].append({
                "id": self.id,
                "delayed_rest": delayed_rest_time,
                "hourly_wage": net_profit / actual_shift_length
            })

class Rider:
    def __init__(self, env, dispatcher, rider_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = rider_id
        
        self.origin = get_rider_origin()
        self.destination = get_rider_destination()
        self.request_time = env.now
        self.matched_event = env.event()
        env.process(self.run())

    def run(self):
        self.dispatcher.waiting_riders.append(self)
        self.dispatcher.match()
        
        patience_time = random.expovariate(PATIENCE_RATE)
        results = yield self.matched_event | self.env.timeout(patience_time)
        
        if self.matched_event not in results:
            if self in self.dispatcher.waiting_riders:
                self.dispatcher.waiting_riders.remove(self)
            self.dispatcher.logs["abandoned_trips"] += 1

def driver_generator(env, dispatcher):
    """Continuously generates drivers based on arrival rate."""
    d_id = 0
    while True:
        yield env.timeout(random.expovariate(DRIVER_ARRIVAL))
        d_id += 1
        Driver(env, dispatcher, d_id)

def rider_generator(env, dispatcher):
    """Continuously generates riders based on arrival rate."""
    r_id = 0
    while True:
        yield env.timeout(random.expovariate(RIDER_ARRIVAL))
        r_id += 1
        Rider(env, dispatcher, r_id)

# ==========================================
# 6. Simulation Replications (Batch Processing)
# ==========================================
NUM_REPLICATIONS = 15
SIM_TIME = 720 # 30 Days in hours

rep_total_requests = []
rep_abandon_rates = []
rep_wait_times = []
rep_hourly_wages = []
rep_wage_stds = []
rep_delayed_rests = []

print(f"Starting {NUM_REPLICATIONS} replications for Phase 3 Interventions Model...")

for i in range(NUM_REPLICATIONS):
    print(f"Running Replication {i+1}/{NUM_REPLICATIONS}...")
    
    # Maintain reproducibility
    random.seed(42 + i)
    np.random.seed(42 + i)
    
    env = simpy.Environment()
    dispatcher = Dispatcher(env)
    
    env.process(driver_generator(env, dispatcher))
    env.process(rider_generator(env, dispatcher))
    
    env.run(until=SIM_TIME)
    
    # KPI Logic
    logs = dispatcher.logs
    total_req = logs['completed_trips'] + logs['abandoned_trips']
    ab_rate = logs['abandoned_trips'] / total_req if total_req > 0 else 0
    # Convert wait time to minutes
    wait_time = (sum(logs['rider_wait_times']) / len(logs['rider_wait_times'])) * 60 if logs['rider_wait_times'] else 0
    
    driver_stats = logs['driver_stats']
    h_wage = np.mean([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
    w_std = np.std([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
    # Convert delayed rest to minutes
    d_rest = np.mean([d['delayed_rest'] for d in driver_stats]) * 60 if driver_stats else 0
    
    rep_total_requests.append(total_req)
    rep_abandon_rates.append(ab_rate)
    rep_wait_times.append(wait_time)
    rep_hourly_wages.append(h_wage)
    rep_wage_stds.append(w_std)
    rep_delayed_rests.append(d_rest)

# ==========================================
# 7. Final Results Reporting
# ==========================================
print("\n" + "="*50)
print(f" PHASE 3 INTERVENTIONS RESULTS (Mean of {NUM_REPLICATIONS} Reps) ")
print("="*50)
print("--- RIDER SATISFACTION ---")
print(f"Total Rider Requests:  {np.mean(rep_total_requests):.0f} (Std: ±{np.std(rep_total_requests):.0f})")
print(f"Abandonment Rate:      {np.mean(rep_abandon_rates):.2%} (Std: ±{np.std(rep_abandon_rates):.2%})")
print(f"Average Wait Time:     {np.mean(rep_wait_times):.2f} mins (Std: ±{np.std(rep_wait_times):.2f})")

print("\n--- DRIVER SATISFACTION ---")
print(f"Average Hourly Wage:   £{np.mean(rep_hourly_wages):.2f} / hr (Std: ±£{np.std(rep_hourly_wages):.2f})")
print(f"Wage Std Dev:          £{np.mean(rep_wage_stds):.2f} (Fairness Check)")
print(f"Avg Delayed Rest Time: {np.mean(rep_delayed_rests):.2f} mins (Std: ±{np.std(rep_delayed_rests):.2f})")
print("="*50)