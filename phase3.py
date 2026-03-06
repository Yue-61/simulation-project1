import simpy
import random
import math
import numpy as np
from scipy.stats import truncnorm

# ==========================================
# 1. Phase 2.5 校准后的真实系统参数 (Table 12)
# ==========================================
MAP_SIZE = 20
SPEED = 20.6052
DRIVER_ARRIVAL = 4.74    # 真实的司机到达率
RIDER_ARRIVAL = 34.31    # 真实的乘客到达率
PATIENCE_RATE = 5.0      # 乘客耐心 (不变)

BASE_FARE = 3.0
FARE_PER_MILE = 2.0
COST_PER_MILE = 0.20

# ==========================================
# 2. 核心干预策略参数 (Phase 3 Interventions)
# ==========================================
# Intervention A: 下班保护机制
PROTECTION_BUFFER = 0.5  # 提前 30 分钟 (0.5小时) 停止接新单，避免强制加班

# Intervention B: 热点区域引导 (Hub Repositioning)
CITY_CENTER = (11.23, 13.26) # 根据组员测出的平均目的地设定为热点中心
REPOSITION_THRESHOLD = 6.0   # 如果送客到了距离中心 6 miles 外的偏远郊区
REPOSITION_DISTANCE = 3.0    # 自动向市中心方向空驶 3 miles 以重新定位

# ==========================================
# 3. 辅助分布函数 (根据组员 Table 8,9,10 拟合)
# ==========================================
def get_truncnorm(mean, std, clip_a=0, clip_b=20):
    """生成 0-20 范围内的截断正态分布坐标"""
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)

def get_rider_origin():
    x = get_truncnorm(7.98, 4.93)
    y = get_truncnorm(12.70, 4.97)
    return (x, y)

def get_rider_destination():
    x = get_truncnorm(11.23, 4.54)
    y = get_truncnorm(13.26, 4.17)
    return (x, y)

def get_driver_initial():
    x = get_truncnorm(9.97, 4.36)
    y = get_truncnorm(11.51, 4.34)
    return (x, y)

def calc_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calc_trip_time(dist):
    """基于真实的行程时间比率 Uniform(0.7, 1.3)"""
    if dist <= 0: return 0.0
    base_time = dist / SPEED
    ratio = random.uniform(0.7, 1.3)
    return base_time * ratio

# ==========================================
# 4. 调度器逻辑
# ==========================================
class Dispatcher:
    def __init__(self, env):
        self.env = env
        self.idle_drivers = []
        self.waiting_riders = []
        self.logs = {
            "completed_trips": 0, "abandoned_trips": 0, 
            "rider_wait_times": [], "driver_stats": []
        }

    def match(self):
        if not self.idle_drivers or not self.waiting_riders:
            return 

        for rider in list(self.waiting_riders):
            if not self.idle_drivers:
                break
            
            # 贪心匹配最近的司机
            closest_driver = min(self.idle_drivers, key=lambda d: calc_distance(d.location, rider.origin))
            
            self.idle_drivers.remove(closest_driver)
            self.waiting_riders.remove(rider)
            
            rider.matched_event.succeed(value=closest_driver)
            closest_driver.assigned_event.succeed(value=rider)

# ==========================================
# 5. 实体类: Driver & Rider (包含 Phase 3 优化)
# ==========================================
class Driver:
    def __init__(self, env, dispatcher, driver_id):
        self.env = env
        self.dispatcher = dispatcher
        self.id = driver_id
        self.location = get_driver_initial()
        
        self.start_time = env.now
        self.planned_shift = random.uniform(6, 8) # 真实的在线时长分布
        self.planned_offline_time = self.start_time + self.planned_shift
        
        self.total_revenue = 0.0
        self.total_distance_driven = 0.0
        self.trips_completed = 0
        
        self.assigned_event = None
        env.process(self.run())

    def run(self):
        # 🟢 PHASE 3 干预 1：下班保护机制
        # 司机只在距离计划下班时间大于 PROTECTION_BUFFER (0.5小时) 时接新单
        while self.env.now < (self.planned_offline_time - PROTECTION_BUFFER):
            self.assigned_event = self.env.event()
            self.dispatcher.idle_drivers.append(self)
            self.dispatcher.match()
            
            time_left = (self.planned_offline_time - PROTECTION_BUFFER) - self.env.now
            if time_left <= 0:
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
            results = yield self.assigned_event | self.env.timeout(time_left)
            
            if self.assigned_event in results:
                rider = self.assigned_event.value
                
                # 接乘客
                dist_to_pickup = calc_distance(self.location, rider.origin)
                self.total_distance_driven += dist_to_pickup
                yield self.env.timeout(calc_trip_time(dist_to_pickup))
                
                wait_time = self.env.now - rider.request_time
                self.dispatcher.logs["rider_wait_times"].append(wait_time)
                
                # 送乘客
                dist_to_dropoff = calc_distance(rider.origin, rider.destination)
                self.total_distance_driven += dist_to_dropoff
                yield self.env.timeout(calc_trip_time(dist_to_dropoff))
                
                trip_revenue = BASE_FARE + (FARE_PER_MILE * dist_to_dropoff)
                self.total_revenue += trip_revenue
                
                self.location = rider.destination
                self.trips_completed += 1
                self.dispatcher.logs["completed_trips"] += 1

                # 🟢 PHASE 3 干预 2：热点区域引导 (Hub Repositioning)
                # 检查是否深处偏远郊区
                dist_to_center = calc_distance(self.location, CITY_CENTER)
                if dist_to_center > REPOSITION_THRESHOLD:
                    # 主动向市中心移动 REPOSITION_DISTANCE 英里
                    move_ratio = REPOSITION_DISTANCE / dist_to_center
                    new_x = self.location[0] + move_ratio * (CITY_CENTER[0] - self.location[0])
                    new_y = self.location[1] + move_ratio * (CITY_CENTER[1] - self.location[1])
                    
                    self.location = (new_x, new_y)
                    self.total_distance_driven += REPOSITION_DISTANCE
                    # 模拟回城的耗时
                    yield self.env.timeout(calc_trip_time(REPOSITION_DISTANCE))
            else:
                if self in self.dispatcher.idle_drivers:
                    self.dispatcher.idle_drivers.remove(self)
                break
                
        # 司机下班结算
        actual_offline_time = self.env.now
        actual_shift_length = actual_offline_time - self.start_time
        # 由于我们加了下班保护，这里的 delayed_rest 会断崖式下降
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
        
        if self.matched_event in results:
            pass
        else:
            if self in self.dispatcher.waiting_riders:
                self.dispatcher.waiting_riders.remove(self)
            self.dispatcher.logs["abandoned_trips"] += 1

# ==========================================
# 6. 运行仿真并打印 KPI
# ==========================================
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

print("Starting Phase 3 Interventions Simulation...")
env = simpy.Environment()
dispatcher = Dispatcher(env)

env.process(driver_generator(env, dispatcher))
env.process(rider_generator(env, dispatcher))

SIM_TIME = 720 # 运行 30 天
env.run(until=SIM_TIME)

# ==========================================
# 7. 输出对比结果
# ==========================================
logs = dispatcher.logs
total_requests = logs["completed_trips"] + logs["abandoned_trips"]
abandon_rate = logs["abandoned_trips"] / total_requests if total_requests > 0 else 0
avg_wait_mins = (sum(logs["rider_wait_times"]) / len(logs["rider_wait_times"])) * 60 if logs["rider_wait_times"] else 0

driver_stats = logs["driver_stats"]
avg_wage = np.mean([d["hourly_wage"] for d in driver_stats]) if driver_stats else 0
wage_std = np.std([d["hourly_wage"] for d in driver_stats]) if driver_stats else 0
avg_delayed_rest = np.mean([d["delayed_rest"] for d in driver_stats]) * 60 if driver_stats else 0

print(f"\n" + "="*45)
print(f" PHASE 3: INTERVENTIONS RESULTS (30 Days) ")
print(f"="*45)
print(f"--- RIDER SATISFACTION ---")
print(f"Total Rider Requests:  {total_requests}")
print(f"Abandonment Rate:      {abandon_rate:.2%}")
print(f"Average Wait Time:     {avg_wait_mins:.2f} mins")

print(f"\n--- DRIVER SATISFACTION ---")
print(f"Average Hourly Wage:   £{avg_wage:.2f} / hr")
print(f"Wage Std Dev:          £{wage_std:.2f} (Fairness Check)")
print(f"Avg Delayed Rest Time: {avg_delayed_rest:.2f} mins")
print(f"="*45)