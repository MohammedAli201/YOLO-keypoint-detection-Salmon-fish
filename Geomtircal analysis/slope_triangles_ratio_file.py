
import ast
import pandas as pd
import math
import json
import os
import numpy as np


import math


import math


import math

import math
keypoints = np. array([[216, 589],
                       [3073, 672],
                       [3073, 703],
                       [500, 608],
                       [604, 447],
                       [784, 701],
                       [716, 821],
                       [801, 812],
                       [1542, 930],
                       [1662, 455],
                       [1639, 928],
                       [2197, 883],
                       [2358, 828],
                       [2318, 549],
                       [2482, 596],
                       [2690, 618],
                       [2815, 580],
                       [2948, 757],
                       [2813, 843],
                       [2687, 802]])
"Thin"
# keypoints = np.array([[39, 48],
#                      [76, 83],
#                       [39, 71],
#                       [73, 57],
#                       [117, 0],
#                       [168, 86],
#                       [158, 114],
#                       [170, 110],
#                       [303, 21],
#                       [398, 33],
#                       [373, 148],
#                       [504, 136],
#                       [541, 123],
#                       [540, 53],
#                       [574, 60],
#                       [621, 72],
#                       [657, 58],
#                       [673, 96],
#                       [645, 127],
#                       [617, 117]])

"jaw deformity /content/Jaw_de_F1-removebg-preview.png"
# keypoints = np.array([[38.556,      166.28],
#                       [67.632,      194.78],
#                       [37.437,      185.24],
#                       [68.38,      172.03],
#                       [110.63,      120.27],
#                       [155.84,      187.65],
#                       [139.94,         220],
#                       [157.74,      218.18],
#                       [341.67,       73.22],
#                       [438.36,       81.33],
#                       [407.71,         233],
#                       [569.9,      208.65],
#                       [619.34,      188.21],
#                       [617.03,      105.45],
#                       [655.22,      117.37],
#                       [692.34,      129.75],
#                       [723.19,      114.58],
#                       [743.16,      156.85],
#                       [717.91,      192.53],
#                       [688.26,      179.08]])
#
# "jaw deformity /content/Jaw_de_F2-removebg-preview.png"
# keypoints = np.array([[39, 159],
#                       [72, 181],
#                       [40, 173],
#                       [73, 161],
#                       [121, 113],
#                       [153, 179],
#                       [138, 214],
#                       [153, 208],
#                       [331,  67],
#                       [430,  81],
#                       [409, 235],
#                       [569, 210],
#                       [611, 194],
#                       [598, 109],
#                       [635, 121],
#                       [691, 140],
#                       [734, 126],
#                       [774, 163],
#                       [734, 196],
#                       [693, 182]])

"""spine deformity d3 detected usin rcnn with black background"""
keypoints = np.array([[35,  99],
                      [62, 129],
                      [34, 119],
                      [63, 108],
                      [110,  67],
                      [140, 132],
                      [121, 170],
                      [140, 166],
                      [311,  29],
                      [407,  41],
                      [381, 206],
                      [515, 185],
                      [556, 170],
                      [547,  80],
                      [571,  96],
                      [595, 100],
                      [617,  86],
                      [655, 118],
                      [386, 211],
                      [506, 183]])

# Test image from test dataset  /content/2022-04-11-21-50-52_right.jpg_1.png
#
# [[36.05551275 52.49761899 28.44292531 26.40075756 46.57252409 55.86591089]]
# keypoints = np.array([[32,  72],
#                       [58,  91],
#                       [33,  84],
#                       [64,  77],
#                       [101,  44],
#                       [131,  88],
#                       [115, 115],
#                       [128, 112],
#                       [263,  25],
#                       [332,  33],
#                       [300, 128],
#                       [412, 116],
#                       [446, 109],
#                       [449,  59],
#                       [472,  65],
#                       [506,  75],
#                       [534,  70],
#                       [540,  98],
#                       [518, 120],
#                       [494, 109]])


# # body_2
keypoints = np.array([[30, 104],
                      [66, 127],
                      [30, 125],
                      [70, 108],
                      [109,  72],
                      [137, 126],
                      [120, 149],
                      [135, 147],
                      [241,  44],
                      [304,  46],
                      [295, 154],
                      [304, 151],
                      [396, 132],
                      [437,  75],
                      [425,  72],
                      [455,  83],
                      [468,  76],
                      [484, 105],
                      [462, 122],
                      [447, 118]])


def process_keypoints(keypoints_list):
    # triangle_indices = [(1, 4, 6), (0, 1, 2)]
    # # Adjusted to have three pairs for each triangle
    # slope_distance = [(1, 9), (4, 10), (6, 8), (0, 3), (1, 3), (2, 3)]
    "These are used for jaw angles and slopes"
    # triangle_indices = [(0, 1, 2), (1, 4, 7)]
    # slope_distance = [(0, 4), (1, 5), (2, 3), (1, 8), (4, 10), (7, 9)]
    "******************************"
    " these are used for spine angles and slopes"
    triangle_indices = [(8, 9, 10), (9, 11, 13)]
    slope_distance = [(8, 7), (9, 1), (10, 4), (9, 17), (11, 14), (13, 19)]
    keypoints = keypoints_list
    # keypoints = [(keypoints_list[i], keypoints_list[i + 1])
    #              for i in range(0, len(keypoints_list), 3)]

    def angle_using_law_of_cosines(a, b, c):
        return math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
    # Helper functions

    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def calculate_slope(p1, p2):
        return (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')

    # Initialize dictionaries to store results
    results = {}
    tr1 = []
    tr2 = []
    sloptr = []

    # Process triangles
    for idx, tri in enumerate(triangle_indices, start=1):
        p1, p2, p3 = keypoints[tri[0]], keypoints[tri[1]], keypoints[tri[2]]
        a = distance(p1, p2)
        b = distance(p2, p3)
        c = distance(p3, p1)

        alpha = angle_using_law_of_cosines(a, b, c)
        beta = angle_using_law_of_cosines(b, c, a)
        gamma = angle_using_law_of_cosines(c, a, b)
        ratios = [a / b, b / c, c / a]
        results[f'Triangle_{idx}'] = {'angles': (
            alpha, beta, gamma), 'ratios': (a / b, b / c, c / a)}

        # For each triangle, process three slopes
        slopes = []
        for j in range(3):
            s_idx = (idx - 1) * 3 + j  # Index for slope_distance
            s1, s2 = keypoints[tri[j]], keypoints[slope_distance[s_idx][1]]
            print(slope_distance[s_idx][1])
            print([tri[j]])
            slope = calculate_slope(s1, s2)
            slopes.append(slope)

        results[f'Triangle_{idx}']['Slopes'] = slopes

    return results


def extract_stats(data, stat_type, triangle, index):
    """Extract statistical values for a specific triangle and index."""
    angle_key = f'{stat_type}_tr_{triangle[-1]}_angle_{index+1}'
    slope_key = f'{stat_type}_tr_{triangle[-1]}_slope_{index+1}'
    ratios_key = f'{stat_type}_tr_{triangle[-1]}_ratio_{index+1}'
    return data[angle_key].values[0], data[slope_key].values[0], data[ratios_key].values[0]


def check_if_within_range(value, mean, std):
    """Check if the value is within the specified range based on mean and standard deviation."""
    return mean - 2 * std <= value <= mean + 2 * std


# Process keypoints and print results
results = process_keypoints(keypoints)
print(results)

# Read CSV data
# data = pd.read_csv('jaw_angles_slopes_ratio.csv')
data = pd.read_csv('spine_angles_slopes_ratio.csv')

# Check all angles and slopes for both triangles
range_check_results = {'Triangle_1': {
    'angles': [], 'slopes': [], 'ratios': []}, 'Triangle_2': {'angles': [], 'slopes': [], 'ratios': []}}

for triangle in ['Triangle_1', 'Triangle_2']:
    for i in range(3):
        angle_mean, slope_mean, ratio_mean = extract_stats(
            data, 'mean', triangle, i)
        angle_std, slope_std, ratio_std = extract_stats(
            data, 'std', triangle, i)

        angle_within_range = check_if_within_range(
            results[triangle]['angles'][i], angle_mean, angle_std)
        slope_within_range = check_if_within_range(
            results[triangle]['Slopes'][i], slope_mean, slope_std)
        ratio_within_range = check_if_within_range(
            results[triangle]['ratios'][i], ratio_mean, ratio_std)

        range_check_results[triangle]['angles'].append(angle_within_range)
        range_check_results[triangle]['slopes'].append(slope_within_range)
        range_check_results[triangle]['ratios'].append(ratio_within_range)

# Print results
for triangle, results in range_check_results.items():
    print(f'{triangle} angles within range: {results["angles"]}')
    print(f'{triangle} slopes within range: {results["slopes"]}')
    print(f'{triangle} ratios within range: {results["ratios"]}')
