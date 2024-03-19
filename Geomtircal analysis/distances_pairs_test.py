import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import math
import numpy as np


# # Thin deformity
# keypoints = [[54.465,     218.71],
#              [71.158, 237.08],
#              [54.609,   228.12],
#              [75.701,    225.59],
#              [112.84,     192.5],
#              [139.19,    236.39],
#              [124.8,    258.81],
#              [136.43,   258.07],
#              [259.93,   175.47],
#              [317.83,    182.54],
#              [289.81,    274.98],
#              [392.55,    263.96],
#              [425.26,    252.98],
#              [429.62,    203.54],
#              [454.59,    207.21],
#              [485.97,    214.76],
#              [510.2,    202.86],
#              [524.43,    230.03],
#              [505.73,     251.75],
#              [482.8,     243.27]]

# # big fish
# keypoints = [[62.612,    255.5],
#              [77.536,    277.02],
#              [63.721,    267.92],
#              [74.199,    258.36],
#              [101.71,    226.89],
#              [127.66,      276.9],
#              [109.03,    301.88],
#              [122.36,      300.4],
#              [234.72,    188.34],
#              [302.46,    192.95],
#              [279.21,     321.27],
#              [398.13,     298.06],
#              [433.59,    279.62],
#              [429.38,     208.16],
#              [457.93,    211.84],
#              [484.68,     216.57],
#              [505.6,     202.48],
#              [518.44,     235.12],
#              [504.16,      266.5],
#              [482.77,     257.79]]

keypoints = np.array([[62.367,      263.65],
                      [78.592,      278.36],
                      [63.909,      275.66],
                      [67.074,      260.19],
                      [91.499,      232.72],
                      [111.8,       277.6],
                      [104.36,       302.2],
                      [113.37,      296.89],
                      [222.15,      184.34],
                      [285.15,      187.69],
                      [277.22,      318.23],
                      [395.27,      293.19],
                      [441.25,      276.46],
                      [424.07,      200.68],
                      [462.81,      210.46],
                      [476.26,       213.2],
                      [0,           0],
                      [0,           0],
                      [498.06,      266.92],
                      [475.21,       260.2]])


keypoints = np.array([[39, 159],
                      [72, 181],
                      [40, 173],
                      [73, 161],
                      [121, 113],
                      [153, 179],
                      [138, 214],
                      [153, 208],
                      [331,  67],
                      [430,  81],
                      [409, 235],
                      [569, 210],
                      [611, 194],
                      [598, 109],
                      [635, 121],
                      [691, 140],
                      [734, 126],
                      [774, 163],
                      [734, 196],
                      [693, 182]])

"body 2"
keypoints = np.array([[30, 113],
                      [63, 133],
                      [29, 123],
                      [67, 114],
                      [102,  79],
                      [128, 123],
                      [114, 150],
                      [128, 147],
                      [239,  44],
                      [306,  45],
                      [302, 153],
                      [306, 154],
                      [405, 128],
                      [412,  66],
                      [432,  73],
                      [461,  80],
                      [470,  75],
                      [488, 108],
                      [465, 126],
                      [450, 116]])

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


def process_keypoints(keypoints_list):
    keypoints = [(keypoints_list[i], keypoints_list[i+1])
                 for i in range(0, len(keypoints_list), 3)]

    return keypoints


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


slope_list = []


# def calculate_distances(datasets, distance_indices):
#     distances = []
#     slope_list_lsit = []
#     for dataset in datasets:
#         distance_set = []
#         for idx, index_pair in enumerate(distance_indices):

#             p1, p2 = dataset[index_pair[0]], dataset[index_pair[1]]
#             if idx == 0:
#                 distance_set.append(distance(p1, p2))
#             else:
#                 temp_point = p1[0], p2[1]
#                 distance_set.append(distance(p1, temp_point))
#         distances.append(distance_set)
#     return distances

def calculate_distances(datasets, distance_indices):
    distances = []

    for dataset in datasets:
        distance_set = []
        for idx, index_pair in enumerate(distance_indices):
            # If by "ratio" you are referring to the slope (or gradient) formed by the line that connects the two points, it would be calculated as:

            p1, p2 = dataset[index_pair[0]], dataset[index_pair[1]]

            distance_set.append(distance(p1, p2))
            # else:
            #     temp_point = p1[0], p2[1]
            #     distance_set.append(distance(p1, temp_point))
        distances.append(distance_set)

    return pd.DataFrame(distances)


# keypoints =keypoints
# distance_indices = [(0, 2), (1, 3), (0, 1), (2, 3), (0, 3), (2, 1)]
# distance_indices = [(15, 19), (16, 18), (15, 16), (19, 18), (15, 18), (19, 16)]
distance_indices = [(15, 19), (16, 18), (15, 16), (19, 18), (15, 18), (19, 16)]

datasets = [keypoints]

mean_list = []
list_distance_std = []
list_distance_mean = []

outside = []
# baseline_data = pd.read_csv('thin_thickness_data.csv')
# print(baseline_data.columns)


def file_load(distances, option=0):
    baseline_data = pd.read_csv('spine_distances.csv')

    distances = np.array(distances)

    # if distances[0][0] < 450:
    #     option = 1
    #     baseline = pd.read_csv('length_450.csv')
    # elif distances[0][0] > 450 and distances[0][0] < 500:
    #     option = 2
    #     baseline = pd.read_csv('length_500.csv')
    # elif distances[0][0] > 500 and distances[0][0] < 550:
    #     option = 3
    #     baseline = pd.read_csv('length_550.csv')
    # elif distances[0][0] > 550 and distances[0][0] < 600:
    #     option = 4
    #     baseline = pd.read_csv('length_600.csv')
    # elif distances[0][0] > 600 and distances[0][0] < 900:
    #     option = 4
    #     baseline = pd.read_csv('length_900.csv')

    return deformity_500(baseline_data, option)


def deformity_500(baseline, option=0):

    for index, row in baseline.iterrows():
        # baseline = row['Baseline']
        cl_names = ['Length', 'Width  1', 'width 2', 'ratio1', 'mean', 'stand', 'ratio2',
                    'mean2', 'stand2']
        mean_ratio1 = row[cl_names[4]]

        std_ratio1 = row[cl_names[5]]

        mean_ratio2 = row[cl_names[7]]

        std_ratio2 = row[cl_names[8]]

        # find the lower and upper limit
        lower_limit1 = mean_ratio1 - 2 * std_ratio1
        upper_limit1 = mean_ratio1 + 2 * std_ratio1

        lower_limit2 = mean_ratio2 - 2 * std_ratio2
        upper_limit2 = mean_ratio2 + 2 * std_ratio2

        print(f'Lower limit1: {lower_limit1} and upper limit1: {upper_limit1}')
        print(f'Lower limit2: {lower_limit2} and upper limit2: {upper_limit2}')

        length_width_ratio_1 = distances[0][0]/distances[0][1]
        length_width_ratio_2 = distances[0][0]/distances[0][2]

        if length_width_ratio_1 < lower_limit1 or length_width_ratio_1 > upper_limit1:
            outside.append(index)
            print(
                f"Deformity found in length_width_ratio_1 . {length_width_ratio_1}    ")
            # print(
            #     f"Test mean at index {index} is outside 1*Std from the baseline mean for ratio 1.")

        if length_width_ratio_2 < lower_limit2 or length_width_ratio_2 > upper_limit2:
            outside.append(index)
            print(
                f"Deformity found in length_width_ratio_2 . {length_width_ratio_2}")
            # print(
            #     f"Test mean at index {index} is outside 1*Std from the baseline mean for ratio 2.")
        else:
            print(f"Deformity not found in this fish ")

        if index == 0:
            break
    return outside


distances = calculate_distances(datasets, distance_indices)
distances = np.array(distances)
print(distances)
# file_load(distances)

# filtered_df = baseline[baseline['Length'] < 490]

# from sklearn.cluster import KMeans
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset
# df = pd.read_csv('distances_ratio.csv')

# # Assuming you have a full dataset
# # Make sure to use the correct column names with the right amount of spaces
# X = df[['Length', 'Width  1', 'Width  2', 'Width 3', 'ratio1', 'ratio2', 'ratio3']]

# # Perform K-Means clustering
# kmeans = KMeans(n_clusters=4)  # for example, choosing 3 clusters
# df['Cluster'] = kmeans.fit_predict(X)
# # Print out centroids
# print(kmeans.cluster_centers_)
# # Now we use the correct column names for the scatterplot
# # Use 'Width  1' with two spaces, matching the DataFrame columns
# sns.scatterplot(data=df, x='Length', y='Width  2',
#                 hue='Cluster', palette='viridis')
# plt.title('Cluster Visualization of Length and Width 2')
# plt.show()
