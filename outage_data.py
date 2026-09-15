import pandas as pd
import numpy as np

tower_hourly = pd.read_csv("tower_hourly_MT_321.csv")

# create time features
tower_hourly["hour"] = pd.to_datetime(tower_hourly["timestamp"]).dt.hour
tower_hourly["day_of_week"] = pd.to_datetime(tower_hourly["timestamp"]).dt.dayofweek

# realistic outage pattern (evening + random)
tower_hourly["outage"] = tower_hourly["hour"].apply(
    lambda x: 1 if x in [18,19,20] else 0
)

# add randomness
tower_hourly["outage"] = tower_hourly["outage"] | np.random.choice([0,1], size=len(tower_hourly), p=[0.9,0.1])

tower_hourly.to_csv("tower_hourly_MT_321_out.csv", index=False)