import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import math
import numpy as np


# # Thin deformity
keypoints = [[54.465,     218.71],
             [71.158, 237.08],
             [54.609,   228.12],
             [75.701,    225.59],
             [112.84,     192.5],
             [139.19,    236.39],
             [124.8,    258.81],
             [136.43,   258.07],
             [259.93,   175.47],
             [317.83,    182.54],
             [289.81,    274.98],
             [392.55,    263.96],
             [425.26,    252.98],
             [429.62,    203.54],
             [454.59,    207.21],
             [485.97,    214.76],
             [510.2,    202.86],
             [524.43,    230.03],
             [505.73,     251.75],
             [482.8,     243.27]]

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

# Thin_f4
# keypoints = np.array([[19.373,      178.67],
#                       [50.643,      202.22],
#                       [17.217,      194.69],
#                       [61.189,      183.14],
#                       [106,      124.18],
#                       [159.16,      193.05],
#                       [131.89,      233.16],
#                       [156.59,      228.56],
#                       [340.58,      80.553],
#                       [437.18,      87.878],
#                       [401.22,      240.94],
#                       [559.53,      215.62],
#                       [606.77,      195.99],
#                       [609.91,      112.81],
#                       [647.1,         124],
#                       [690.14,      136.74],
#                       [723.23,      119.61],
#                       [745.5,      163.86],
#                       [715.22,      199.12],
#                       [682.97,      185.19]])

# Thin_f5
# keypoints = np.array([[29.136,      154.39],
#                       [60.919,      182.07],
#                       [35.58,      168.24],
#                       [66.93,      162.69],
#                       [104.31,      111.96],
#                       [140.66,      174.34],
#                       [129.13,      204.94],
#                       [148.46,      202.69],
#                       [291.88,      70.473],
#                       [373.74,      74.997],
#                       [345.24,      210.99],
#                       [506.31,      184.15],
#                       [557.52,       173.3],
#                       [559.86,      95.201],
#                       [602.11,      105.29],
#                       [667.27,      111.46],
#                       [710.14,      96.103],
#                       [740.76,      139.07],
#                       [708.55,      174.71],
#                       [667.8,      162.21]])

# # Thin_f5 using rcnn with smoothing edges

keypoints = np.array([[40, 133],
                      [69, 159],
                      [38, 154],
                      [72, 138],
                      [127,  93],
                      [153, 165],
                      [141, 192],
                      [139, 192],
                      [311,  81],
                      [414,  91],
                      [401, 205],
                      [556, 183],
                      [571, 180],
                      [581,  98],
                      [598, 103],
                      [668, 107],
                      [696,  94],
                      [735, 130],
                      [696, 169],
                      [668, 161]])

# Thin_f1
# keypoints = np.array([[33.187,      143.86],
#                       [61.599,      169.07],
#                       [37.342,      157.87],
#                       [74.89,      151.56],
#                       [125.34,      104.01],
#                       [156.76,      165.36],
#                       [147.34,      193.95],
#                       [162.23,      193.36],
#                       [318.21,      76.608],
#                       [388.97,      81.979],
#                       [358.64,      203.07],
#                       [502.98,      187.15],
#                       [550.09,      178.86],
#                       [556.8,      112.01],
#                       [594.44,      124.18],
#                       [648.13,      132.67],
#                       [689.68,      118.68],
#                       [715.57,       156.5],
#                       [681.15,       186.8],
#                       [653.29,      175.54]])

# # Thin_f3
keypoints = np.array([[42, 129],
                      [74, 166],
                      [39, 152],
                      [75, 139],
                      [822,  53],
                      [160, 174],
                      [140, 199],
                      [160, 191],
                      [309, 105],
                      [415, 115],
                      [397, 224],
                      [558, 211],
                      [582, 201],
                      [585, 126],
                      [616, 131],
                      [675, 138],
                      [705, 119],
                      [735, 159],
                      [707, 193],
                      [675, 185]])

# # # body_2
# keypoints = np.array([[30, 104],
#                       [66, 127],
#                       [30, 125],
#                       [70, 108],
#                       [109,  72],
#                       [137, 126],
#                       [120, 149],
#                       [135, 147],
#                       [241,  44],
#                       [304,  46],
#                       [295, 154],
#                       [304, 151],
#                       [396, 132],
#                       [437,  75],
#                       [425,  72],
#                       [455,  83],
#                       [468,  76],
#                       [484, 105],
#                       [462, 122],
#                       [447, 118]])


# # # body_3

# keypoints = np.array([[36, 100],
#                       [63, 129],
#                       [36, 115],
#                       [69, 109],
#                       [123,  67],
#                       [151, 142],
#                       [121, 166],
#                       [149, 158],
#                       [316,  26],
#                       [392,  38],
#                       [381, 205],
#                       [382, 207],
#                       [433, 191],
#                       [508,  63],
#                       [559,  96],
#                       [583, 111],
#                       [614,  83],
#                       [691,  62],
#                       [386, 211],
#                       [518, 179]])

"Big fish"
# keypoints = np.array([[17.715,      179.45],
#                       [47.61,      218.14],
#                       [22.131,      202.79],
#                       [40.371,      186.89],
#                       [97.041,      125.72],
#                       [146.49,      213.28],
#                       [126.98,      250.77],
#                       [145.68,      251.82],
#                       [379.37,      52.372],
#                       [492.17,      61.208],
#                       [464.17,      270.81],
#                       [645.34,       227.3],
#                       [702.33,      202.23],
#                       [692.97,       91.74],
#                       [737.65,      106.09],
#                       [769.56,      113.39],
#                       [795.27,         103],
#                       [816.62,      151.44],
#                       [790.44,      197.96],
#                       [765.46,      176.37]])

# keypoints = [[22.901,      81.469],
#              [47.838,      101.36],
#              [25.595,      95.061],
#              [49.954,      86.521],
#              [80.01,      54.602],
#              [109,      98.593],
#              [89.962,      121.42],
#              [105.08,      121.71],
#              [210.9,      23.383],
#              [273.93,      28.071],
#              [248.16,      134.27],
#              [358.94,      117.35],
#              [391.32,      102.89],
#              [391.9,      44.449],
#              [418.67,      46.636],
#              [455.98,      53.796],
#              [480.54,      40.363],
#              [499.16,      70.788],
#              [480.42,      99.044],
#              [451.97,      89.138]]


def process_keypoints(keypoints_list):
    keypoints = [(keypoints_list[i], keypoints_list[i+1])
                 for i in range(0, len(keypoints_list), 3)]

    return keypoints


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


slope_list = []


def calculate_distances(datasets, distance_indices):
    distances = []
    slope_list_lsit = []
    for dataset in datasets:
        distance_set = []
        for idx, index_pair in enumerate(distance_indices):

            p1, p2 = dataset[index_pair[0]], dataset[index_pair[1]]
            if idx == 0:
                distance_set.append(distance(p1, p2))
            else:
                temp_point = p1[0], p2[1]
                distance_set.append(distance(p1, temp_point))
        distances.append(distance_set)
    return distances


# keypoints =keypoints
# distance_indices = [(0, 17),  (8, 10), (16, 18)]
distance_indices = [(0, 17), (4, 6), (8, 10)]
datasets = [keypoints]

mean_list = []
list_distance_std = []
list_distance_mean = []

outside = []
# baseline_data = pd.read_csv('thin_thickness_data.csv')
# print(baseline_data.columns)


def file_load(distances, option=0):
    baseline_data = pd.read_csv('new_thin_thick_test_as_copy.csv')

    distances = np.array(distances)

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

            # print(
            #     f"Deformity found in length_width_ratio_1 . {length_width_ratio_1}    ")
            print(
                f"Deformity found in length_width_ratio_2 . {length_width_ratio_2}")

        else:
            print(f"Deformity not found in this fish ")

        if index == 0:
            break
    return outside


distances = calculate_distances(datasets, distance_indices)
distances = np.array(distances)
file_load(distances)

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
