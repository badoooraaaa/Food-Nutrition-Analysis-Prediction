import pandas as pd
from scipy.stats import norm
import numpy as np

# WHO Child Growth Standards 2006
# Length/height-for-age and Weight-for-age for boys and girls (0-24 months)
# Percentiles: 3rd, 15th, 50th, 85th, 97th

# Weight-for-age BOYS (kg)
weight_for_age_boys_data = [
    [0, 2.5, 2.9, 3.3, 3.8, 4.4],
    [1, 3.4, 3.9, 4.5, 5.1, 5.8],
    [2, 4.3, 4.9, 5.6, 6.3, 7.1],
    [3, 5.0, 5.7, 6.4, 7.2, 8.0],
    [4, 5.6, 6.2, 7.0, 7.8, 8.7],
    [5, 6.0, 6.7, 7.5, 8.4, 9.3],
    [6, 6.4, 7.1, 7.9, 8.8, 9.8],
    [7, 6.7, 7.4, 8.3, 9.2, 10.3],
    [8, 6.9, 7.7, 8.6, 9.6, 10.7],
    [9, 7.1, 8.0, 8.9, 9.9, 11.0],
    [10, 7.4, 8.2, 9.2, 10.2, 11.4],
    [11, 7.6, 8.4, 9.4, 10.5, 11.7],
    [12, 7.7, 8.6, 9.6, 10.8, 12.0],
    [13, 7.9, 8.8, 9.9, 11.0, 12.3],
    [14, 8.1, 9.0, 10.1, 11.3, 12.6],
    [15, 8.3, 9.2, 10.3, 11.5, 12.8],
    [16, 8.4, 9.4, 10.5, 11.7, 13.1],
    [17, 8.6, 9.6, 10.7, 12.0, 13.4],
    [18, 8.8, 9.8, 10.9, 12.2, 13.7],
    [19, 8.9, 10.0, 11.1, 12.5, 13.9],
    [20, 9.1, 10.1, 11.3, 12.7, 14.2],
    [21, 9.2, 10.3, 11.5, 12.9, 14.5],
    [22, 9.4, 10.5, 11.8, 13.2, 14.7],
    [23, 9.5, 10.7, 12.0, 13.4, 15.0],
    [24, 9.7, 10.8, 12.2, 13.6, 15.3]
]
weight_for_age_boys = pd.DataFrame(weight_for_age_boys_data, columns=['age_months', 'P3', 'P15', 'P50', 'P85', 'P97'])

# Weight-for-age GIRLS (kg)
weight_for_age_girls_data = [
    [0, 2.4, 2.8, 3.2, 3.7, 4.2],
    [1, 3.2, 3.6, 4.2, 4.8, 5.5],
    [2, 3.9, 4.5, 5.1, 5.8, 6.6],
    [3, 4.5, 5.2, 5.8, 6.6, 7.5],
    [4, 5.0, 5.7, 6.4, 7.3, 8.2],
    [5, 5.4, 6.1, 6.9, 7.8, 8.8],
    [6, 5.7, 6.5, 7.3, 8.2, 9.3],
    [7, 6.0, 6.8, 7.6, 8.6, 9.8],
    [8, 6.3, 7.0, 7.9, 9.0, 10.2],
    [9, 6.5, 7.3, 8.2, 9.3, 10.5],
    [10, 6.7, 7.5, 8.5, 9.6, 10.9],
    [11, 6.9, 7.7, 8.7, 9.9, 11.2],
    [12, 7.0, 7.9, 8.9, 10.1, 11.5],
    [13, 7.2, 8.1, 9.2, 10.4, 11.8],
    [14, 7.4, 8.3, 9.4, 10.6, 12.1],
    [15, 7.6, 8.5, 9.6, 10.9, 12.4],
    [16, 7.7, 8.7, 9.8, 11.1, 12.6],
    [17, 7.9, 8.9, 10.0, 11.4, 12.9],
    [18, 8.1, 9.1, 10.2, 11.6, 13.2],
    [19, 8.2, 9.2, 10.4, 11.8, 13.5],
    [20, 8.4, 9.4, 10.6, 12.1, 13.7],
    [21, 8.6, 9.6, 10.9, 12.3, 14.0],
    [22, 8.7, 9.8, 11.1, 12.5, 14.3],
    [23, 8.9, 10.0, 11.3, 12.8, 14.6],
    [24, 9.0, 10.2, 11.5, 13.0, 14.8]
]
weight_for_age_girls = pd.DataFrame(weight_for_age_girls_data, columns=['age_months', 'P3', 'P15', 'P50', 'P85', 'P97'])

# Height-for-age BOYS (cm)
height_for_age_boys_data = [
    [0, 46.1, 48.0, 49.9, 51.8, 53.7],
    [1, 50.8, 52.8, 54.7, 56.7, 58.6],
    [2, 54.4, 56.4, 58.4, 60.4, 62.4],
    [3, 57.3, 59.4, 61.4, 63.5, 65.5],
    [4, 59.7, 61.8, 63.9, 66.0, 68.0],
    [5, 61.7, 63.8, 65.9, 68.0, 70.1],
    [6, 63.3, 65.5, 67.6, 69.8, 71.9],
    [7, 64.8, 67.0, 69.2, 71.3, 73.5],
    [8, 66.2, 68.4, 70.6, 72.8, 75.0],
    [9, 67.5, 69.7, 72.0, 74.2, 76.5],
    [10, 68.7, 71.0, 73.3, 75.6, 77.9],
    [11, 69.9, 72.2, 74.5, 76.9, 79.2],
    [12, 71.0, 73.4, 75.7, 78.1, 80.5],
    [13, 72.1, 74.5, 76.9, 79.3, 81.8],
    [14, 73.1, 75.6, 78.0, 80.5, 83.0],
    [15, 74.1, 76.6, 79.1, 81.7, 84.2],
    [16, 75.0, 77.5, 80.2, 82.8, 85.4],
    [17, 76.0, 78.6, 81.2, 83.9, 86.5],
    [18, 76.9, 79.6, 82.3, 85.0, 87.7],
    [19, 77.7, 80.5, 83.2, 86.0, 88.8],
    [20, 78.6, 81.4, 84.2, 87.0, 89.8],
    [21, 79.4, 82.3, 85.1, 88.0, 90.9],
    [22, 80.2, 83.1, 86.0, 89.0, 91.9],
    [23, 81.0, 83.9, 86.9, 89.9, 92.8],
    [24, 81.7, 84.8, 87.8, 90.9, 93.9]
]
height_for_age_boys = pd.DataFrame(height_for_age_boys_data, columns=['age_months', 'P3', 'P15', 'P50', 'P85', 'P97'])

# Height-for-age GIRLS (cm)
height_for_age_girls_data = [
    [0, 45.4, 47.3, 49.1, 51.0, 52.9],
    [1, 49.8, 51.7, 53.7, 55.6, 57.6],
    [2, 53.0, 55.0, 57.1, 59.1, 61.1],
    [3, 55.6, 57.7, 59.8, 61.9, 64.0],
    [4, 57.8, 59.9, 62.1, 64.3, 66.4],
    [5, 59.6, 61.8, 64.0, 66.2, 68.5],
    [6, 61.2, 63.5, 65.7, 68.0, 70.3],
    [7, 62.7, 65.0, 67.3, 69.6, 71.9],
    [8, 64.0, 66.4, 68.7, 71.1, 73.5],
    [9, 65.3, 67.7, 70.1, 72.6, 75.0],
    [10, 66.5, 69.0, 71.5, 73.9, 76.4],
    [11, 67.7, 70.3, 72.8, 75.3, 77.8],
    [12, 68.9, 71.4, 74.0, 76.6, 79.2],
    [13, 70.0, 72.6, 75.2, 77.8, 80.5],
    [14, 71.0, 73.7, 76.4, 79.1, 81.7],
    [15, 72.0, 74.8, 77.5, 80.2, 83.0],
    [16, 73.0, 75.8, 78.6, 81.4, 84.2],
    [17, 74.0, 76.8, 79.7, 82.5, 85.4],
    [18, 74.9, 77.8, 80.7, 83.6, 86.5],
    [19, 75.8, 78.8, 81.7, 84.7, 87.6],
    [20, 76.7, 79.7, 82.7, 85.7, 88.7],
    [21, 77.5, 80.6, 83.7, 86.7, 89.8],
    [22, 78.4, 81.5, 84.6, 87.7, 90.8],
    [23, 79.2, 82.3, 85.5, 88.7, 91.9],
    [24, 80.0, 83.2, 86.4, 89.6, 92.9]
]
height_for_age_girls = pd.DataFrame(height_for_age_girls_data, columns=['age_months', 'P3', 'P15', 'P50', 'P85', 'P97'])

def calculate_percentile(value, age_months, sex, metric):
    """
    Calculates the approximate percentile, z-score, and status 
    based on WHO growth standards data.
    """
    age_months = max(0, min(24, int(round(age_months))))
    sex = sex.lower()
    metric = metric.lower()
    
    if metric == 'weight':
        df = weight_for_age_boys if sex == 'boy' else weight_for_age_girls
    else:
        df = height_for_age_boys if sex == 'boy' else height_for_age_girls
        
    row = df[df['age_months'] == age_months].iloc[0]
    
    # Simple interpolation assuming normal distribution approximation
    # WHO uses LMS, but given we only have percentiles, we'll map Z-scores:
    # P50 -> Z=0
    # P15 -> Z=-1.036
    # P3  -> Z=-1.881
    # P85 -> Z=+1.036
    # P97 -> Z=+1.881
    
    median = row['P50']
    
    # Approximate standard deviation based on whether the value is above or below median
    # We use distance to 15th or 85th percentile (approx 1 SD)
    if value < median:
        sd = (median - row['P15']) / 1.036
    else:
        sd = (row['P85'] - median) / 1.036
        
    # Prevent division by zero
    sd = max(sd, 0.01)
    
    z_score = (value - median) / sd
    
    # Calculate exact percentile based on normal distribution CDF
    percentile = norm.cdf(z_score) * 100
    
    # Determine status
    if percentile < 3:
        status = "Severely underweight" if metric == "weight" else "Severely stunted"
    elif percentile < 15:
        status = "Underweight" if metric == "weight" else "Stunted"
    elif percentile > 97:
        status = "Obese" if metric == "weight" else "Tall"
    elif percentile > 85:
        status = "Overweight" if metric == "weight" else "Above average"
    else:
        status = "Normal"
        
    return float(percentile), float(z_score), status
