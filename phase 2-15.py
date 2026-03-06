import simpy
import random
import math
import numpy as np

# ==========================================
# 1. Phase 2 校准后的真实参数 (Data-Calibrated)
# ==========================================
MAP_SIZE = 20            
SPEED = 20.61            # 真实平均车速
DRIVER_ARRIVAL = 4.74    # 真实司机到达率
RIDER_ARRIVAL = 34.31    # 真实乘客到达率
PATIENCE_RATE = 5.0      

BASE_FARE = 3.0          
FARE_PER_MILE = 2.0      
COST_PER_MILE = 0.20     

# ==========================================
# 2. 辅助函数
# ==========================================
def get_random_location():
    # Phase 2 前半部分（Table 2）依然保持空间均匀分布，为了控制变量
    return (random.uniform(0, MAP_SIZE), random.uniform(0, MAP_SIZE))

def calc_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calc_trip_time(dist):
    if dist <= 0: return 0.0
    base_time = dist / SPEED
    # 使用校准后的真实时间波动
    ratio = random.uniform(0.7, 1.3)
    return base_time * ratio

# ==========================================
# 3. 调度器 (Dispatcher)
# ==========================================
class Dispatcher:
    def __init__(self, env):
        self.env = env
        self.idle_drivers = []   
        self.waiting_riders = [] 
        
        self.logs = {
            'completed_trips': 0, 
            'abandoned_trips': 0, 
            'rider_wait_times': [], 
            'driver_stats': []      
        }

    def match(self):
        if not self.idle_drivers or not self.waiting_riders:
            return 

        for rider in list(self.waiting_riders):
            if not self.idle_drivers:
                break 

            closest_driver = min(self.idle_drivers, key=lambda d: calc_distance(d.location, rider.origin))
            
            self.idle_drivers.remove(closest_driver)
            self.waiting_riders.remove(rider)
            
            rider.matched_event.succeed(value=closest_driver)
            closest_driver.assigned_event.succeed(value=rider)

# ==========================================
# 4. 实体: Driver & Rider
# ==========================================
class Driver:
    def __init__(self, env, dispatcher, driver_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = driver_id
        self.location = get_random_location()
        
        self.start_time = env.now
        self.planned_shift = random.uniform(6, 8) # 使用校准后的真实在线时长
        self.planned_offline_time = self.start_time + self.planned_shift
        
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
                
            results = yield self.assigned_event | self.env.timeout(time_left)
            
            if self.assigned_event in results:
                rider = self.assigned_event.value
                
                dist_to_pickup = calc_distance(self.location, rider.origin)
                self.total_distance_driven += dist_to_pickup
                yield self.env.timeout(calc_trip_time(dist_to_pickup))
                
                wait_time = self.env.now - rider.request_time
                self.dispatcher.logs['rider_wait_times'].append(wait_time)
                
                dist_to_dropoff = calc_distance(rider.origin, rider.destination)
                self.total_distance_driven += dist_to_dropoff
                yield self.env.timeout(calc_trip_time(dist_to_dropoff))
                
                trip_revenue = BASE_FARE + (FARE_PER_MILE * dist_to_dropoff)
                self.total_revenue += trip_revenue
                
                self.location = rider.destination
                self.trips_completed += 1
                self.dispatcher.logs['completed_trips'] += 1
            else:
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
        actual_offline_time = self.env.now
        actual_shift_length = actual_offline_time - self.start_time
        delayed_rest_time = max(0, actual_offline_time - self.planned_offline_time)
        
        total_cost = self.total_distance_driven * COST_PER_MILE
        net_profit = self.total_revenue - total_cost
        
        if actual_shift_length > 0:
            self.dispatcher.logs['driver_stats'].append({
                'id': self.id,
                'delayed_rest': delayed_rest_time,
                'hourly_wage': net_profit / actual_shift_length
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
        self.dispatcher.waiting_riders.append(self)
        self.dispatcher.match()
        
        patience_time = random.expovariate(PATIENCE_RATE)
        results = yield self.matched_event | self.env.timeout(patience_time)
        
        if self.matched_event in results:
            pass 
        else:
            if self in self.dispatcher.waiting_riders:
                self.dispatcher.waiting_riders.remove(self)
            self.dispatcher.logs['abandoned_trips'] += 1

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

# ==========================================
# 5. 运行 15 次循环
# ==========================================
NUM_REPLICATIONS = 15
SIM_TIME = 720 # 30 Days

rep_total_requests = []
rep_abandon_rates = []
rep_wait_times = []
rep_hourly_wages = []
rep_wage_stds = []
rep_delayed_rests = []

print(f"Starting {NUM_REPLICATIONS} replications for Phase 2 Data-Calibrated Model...")

for i in range(NUM_REPLICATIONS):
    print(f"Running Replication {i+1}/{NUM_REPLICATIONS}...")
    
    random.seed(42 + i)
    np.random.seed(42 + i)
    
    env = simpy.Environment()
    dispatcher = Dispatcher(env)
    
    env.process(driver_generator(env, dispatcher))
    env.process(rider_generator(env, dispatcher))
    
    env.run(until=SIM_TIME)
    
    logs = dispatcher.logs
    total_req = logs['completed_trips'] + logs['abandoned_trips']
    ab_rate = logs['abandoned_trips'] / total_req if total_req > 0 else 0
    wait_time = (sum(logs['rider_wait_times']) / len(logs['rider_wait_times'])) * 60 if logs['rider_wait_times'] else 0
    
    driver_stats = logs['driver_stats']
    h_wage = np.mean([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
    w_std = np.std([d['hourly_wage'] for d in driver_stats]) if driver_stats else 0
    d_rest = np.mean([d['delayed_rest'] for d in driver_stats]) * 60 if driver_stats else 0
    
    rep_total_requests.append(total_req)
    rep_abandon_rates.append(ab_rate)
    rep_wait_times.append(wait_time)
    rep_hourly_wages.append(h_wage)
    rep_wage_stds.append(w_std)
    rep_delayed_rests.append(d_rest)

# ==========================================
# 6. 打印最终合并结果
# ==========================================
print("\n" + "="*50)
print(f" PHASE 2 DATA-CALIBRATED RESULTS (Mean of {NUM_REPLICATIONS} Reps) ")
print("="*50)
print(f"Total Rider Requests:  {np.mean(rep_total_requests):.0f} (Std: ±{np.std(rep_total_requests):.0f})")
print(f"Abandonment Rate:      {np.mean(rep_abandon_rates):.2%} (Std: ±{np.std(rep_abandon_rates):.2%})")
print(f"Average Wait Time:     {np.mean(rep_wait_times):.2f} mins (Std: ±{np.std(rep_wait_times):.2f})")
print(f"Average Hourly Wage:   £{np.mean(rep_hourly_wages):.2f} / hr (Std: ±£{np.std(rep_hourly_wages):.2f})")
print(f"Wage Std Dev:          £{np.mean(rep_wage_stds):.2f}")
print(f"Avg Delayed Rest Time: {np.mean(rep_delayed_rests):.2f} mins (Std: ±{np.std(rep_delayed_rests):.2f})")
print("="*50)