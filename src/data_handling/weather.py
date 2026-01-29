from meteostat import Daily, Point
from datetime import datetime
import pandas as pd

locations = {
    "Karnataka": ("Bengaluru", 12.97, 77.59),
    "Madhya Pradesh": ("Bhopal", 23.25, 77.41),
    "Andhra Pradesh": ("Vijayawada", 16.51, 80.64),
    "Uttar Pradesh": ("Lucknow", 26.85, 80.95),
    "West Bengal": ("Kolkata", 22.57, 88.36),
    "Haryana": ("Chandigarh", 30.74, 76.79),
    "Punjab": ("Ludhiana", 30.90, 75.85),
}

start = datetime(2024, 12, 1)
end = datetime(2025, 12, 31)

all_weather = []

for state, (city, lat, lon) in locations.items():
    point = Point(lat, lon)
    data = Daily(point, start, end).fetch()
    data = data.reset_index()
    data['State'] = state
    data['City'] = city
    all_weather.append(data)

weather_df = pd.concat(all_weather)

weather_df = weather_df[['time','State','City','tavg','tmin','tmax','prcp']]
weather_df.columns = ['Date','State','City','Temp_Avg','Temp_Min','Temp_Max','Rainfall']

weather_df.to_csv("state_level_weather.csv", index=False)

print(weather_df.head())
