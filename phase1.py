import simpy
import random
import math
import numpy as np

# 1. Baseline Parameters
MAP_SIZE = 20            # Squareshire is 20x20 miles
SPEED = 20.0             # Average speed is 20 mph
DRIVER_ARRIVAL = 3.0     # Driver arrival rate: Exp(3/hr)
RIDER_ARRIVAL = 30.0     # Rider arrival rate: Exp(30/hr)
PATIENCE_RATE = 5.0      # Rider patience rate: Exp(5/hr)

# Financial and Cost Parameters
BASE_FARE = 3.0          # Initial charge £3
FARE_PER_MILE = 2.0      # £2 per mile (only for distance from origin to destination)
COST_PER_MILE = 0.20     # Driver petrol cost £0.20 per mile (all distance driven)

# 2. Helper Functions
def get_random_location():
    """Generates a uniform random coordinate within the 20x20 map."""
    return (random.uniform(0, MAP_SIZE), random.uniform(0, MAP_SIZE))

def calc_distance(loc1, loc2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calc_trip_time(dist):
    """Calculates actual trip time: Uniform(0.8*mu, 1.2*mu)."""
    if dist == 0: return 0
    mu_t = dist / SPEED
    return random.uniform(0.8 * mu_t, 1.2 * mu_t)

# 3. The Dispatcher (BoxCar App Logic)
class Dispatcher:
    def __init__(self, env):
        self.env = env
        self.idle_drivers = []   
        self.waiting_riders = [] 
        
        # Dictionary to log system-wide KPIs
        self.logs = {
            'completed_trips': 0, 
            'abandoned_trips': 0, 
            'rider_wait_times': [], # Logs wait times in hours
            'driver_stats': []      # Logs KPI data for each driver upon going offline
        }

    def match(self):
        """Core dispatching logic: Matches idle drivers with waiting riders based on closest distance."""
        if not self.idle_drivers or not self.waiting_riders:
            return 

        # Iterate through waiting riders to find the closest available driver
        for rider in list(self.waiting_riders):
            if not self.idle_drivers:
                break # No more drivers available

            # Find the driver with the minimum Euclidean distance to the rider's origin
            closest_driver = min(self.idle_drivers, key=lambda d: calc_distance(d.location, rider.origin))
            
            # Remove them from the availability/waiting pools
            self.idle_drivers.remove(closest_driver)
            self.waiting_riders.remove(rider)
            
            # Trigger events to notify both entities of the successful match
            rider.matched_event.succeed(value=closest_driver)
            closest_driver.assigned_event.succeed(value=rider)

# 4. Entities: Driver & Rider
class Driver:
    def __init__(self, env, dispatcher, driver_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = driver_id
        self.location = get_random_location()
        
        self.start_time = env.now
        self.planned_shift = random.uniform(5, 8) 
        self.planned_offline_time = self.start_time + self.planned_shift
        
        # Driver-specific KPIs
        self.total_revenue = 0.0
        self.total_distance_driven = 0.0
        self.trips_completed = 0
        
        self.assigned_event = None
        env.process(self.run())

    def run(self):
        while self.env.now < self.planned_offline_time:
            self.assigned_event = self.env.event()
            self.dispatcher.idle_drivers.append(self)
            self.dispatcher.match()
            
            time_left = self.planned_offline_time - self.env.now
            if time_left <= 0:
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
            # Wait for either an assignment or for the shift to end
            results = yield self.assigned_event | self.env.timeout(time_left)
            
            if self.assigned_event in results:
                rider = self.assigned_event.value
                
                # 1. Deadhead trip: Drive to pick up the rider
                dist_to_pickup = calc_distance(self.location, rider.origin)
                self.total_distance_driven += dist_to_pickup
                yield self.env.timeout(calc_trip_time(dist_to_pickup))
                
                # Log Rider Wait Time KPI
                wait_time = self.env.now - rider.request_time
                self.dispatcher.logs['rider_wait_times'].append(wait_time)
                
                # 2. Service trip: Drive rider to destination
                dist_to_dropoff = calc_distance(rider.origin, rider.destination)
                self.total_distance_driven += dist_to_dropoff
                yield self.env.timeout(calc_trip_time(dist_to_dropoff))
                
                # 3. Calculate trip revenue and update state
                trip_revenue = BASE_FARE + (FARE_PER_MILE * dist_to_dropoff)
                self.total_revenue += trip_revenue
                
                self.location = rider.destination
                self.trips_completed += 1
                self.dispatcher.logs['completed_trips'] += 1
            else:
                # Shift ended while idle
                self.dispatcher.idle_drivers.remove(self)
                break
                
        # Driver goes offline, calculate final KPIs
        actual_offline_time = self.env.now
        actual_shift_length = actual_offline_time - self.start_time
        delayed_rest_time = max(0, actual_offline_time - self.planned_offline_time)
        
        total_cost = self.total_distance_driven * COST_PER_MILE
        net_profit = self.total_revenue - total_cost
        
        self.dispatcher.logs['driver_stats'].append({
            'id': self.id,
            'shift_length': actual_shift_length,
            'delayed_rest': delayed_rest_time,
            'net_profit': net_profit,
            'trips': self.trips_completed,
            'hourly_wage': net_profit / actual_shift_length if actual_shift_length > 0 else 0
        })

class Rider:
    def __init__(self, env, dispatcher, rider_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = rider_id
        self.origin = get_random_location()
        self.destination = get_random_location()
        self.request_time = env.now
        
        self.matched_event = env.event()
        env.process(self.run())

    def run(self):
        # 1. Join waiting pool and trigger matching
        self.dispatcher.waiting_riders.append(self)
        self.dispatcher.match()
        
        # 2. Patience countdown
        patience_time = random.expovariate(PATIENCE_RATE)
        
        # Competing events: Match vs. Patience Expiration
        results = yield self.matched_event | self.env.timeout(patience_time)
        
        if self.matched_event in results:
            pass # Successfully picked up
        else:
            # Rider lost patience and abandoned the trip
            if self in self.dispatcher.waiting_riders:
                self.dispatcher.waiting_riders.remove(self)
            self.dispatcher.logs['abandoned_trips'] += 1

# 5. Generators & Main Execution
def driver_generator(env, dispatcher):
    d_id = 0
    while True:
        yield env.timeout(random.expovariate(DRIVER_ARRIVAL))
        d_id += 1
        Driver(env, dispatcher, d_id)

def rider_generator(env, dispatcher):
    r_id = 0
    while True:
        yield env.timeout(random.expovariate(RIDER_ARRIVAL))
        r_id += 1
        Rider(env, dispatcher, r_id)

# Run a 30-day simulation (720 hours) to obtain stable data
print("Starting BoxCar Baseline Simulation...")
env = simpy.Environment()
dispatcher = Dispatcher(env)

env.process(driver_generator(env, dispatcher))
env.process(rider_generator(env, dispatcher))

SIM_TIME = 720 
env.run(until=SIM_TIME)

# 6. KPI Summary Report
logs = dispatcher.logs
total_requests = logs['completed_trips'] + logs['abandoned_trips']
abandon_rate = logs['abandoned_trips'] / total_requests if total_requests > 0 else 0
avg_wait_mins = (sum(logs['rider_wait_times']) / len(logs['rider_wait_times'])) * 60 if logs['rider_wait_times'] else 0

driver_stats = logs['driver_stats']
avg_wage = np.mean([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
wage_std = np.std([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
avg_delayed_rest = np.mean([d['delayed_rest'] for d in driver_stats]) * 60 if driver_stats else 0

print(f"\n" + "="*40)
print(f"   BASELINE SIMULATION RESULTS (30 Days)    ")
print(f"="*40)
print(f"--- RIDER SATISFACTION ---")
print(f"Total Rider Requests:  {total_requests}")
print(f"Abandonment Rate:      {abandon_rate:.2%}")
print(f"Average Wait Time:     {avg_wait_mins:.2f} mins")

print(f"\n--- DRIVER SATISFACTION ---")
print(f"Average Hourly Wage:   £{avg_wage:.2f} / hr")
print(f"Wage Std Dev:          £{wage_std:.2f} (Lower = More Fair)")
print(f"Avg Delayed Rest Time: {avg_delayed_rest:.2f} mins")
print(f"="*40)