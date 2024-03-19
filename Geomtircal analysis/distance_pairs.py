import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import numpy as np


def process_keypoints(keypoints_list):
    keypoints = [(keypoints_list[i], keypoints_list[i+1])
                 for i in range(0, len(keypoints_list), 3)]

    return keypoints


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


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


with open('train.json', 'r') as file:
    print("Loading train JSON file...")
    data = json.load(file)

datasets = []
for annotation in data["annotations"]:
    KeyPoints = annotation["keypoints"]
    results = process_keypoints(KeyPoints)
    datasets.append(results)


# distance_indices = [(1, 5), (4, 6)]
# distance_indices = [(0, 17), (4, 6), (8, 10)]
# distance_indices = [(0, 2), (1, 3), (0, 1), (2, 3), (0, 3), (2, 1)]
"""
Spine deformity distances
"""
distance_indices = [(15, 19), (16, 18), (15, 16), (19, 18), (15, 18), (19, 16)]
mean_distances = []
# distances = calculate_distances(datasets, distance_indices)
distances_df = calculate_distances(datasets, distance_indices)
print(distances_df)
# Calculate the ratios
# ratios_df = pd.DataFrame()
"""
Jad deformity properties"
"""

jaw_properties = pd.DataFrame()


# ratios_df['ratio1'] = distances_df.iloc[:, 0] / distances_df.iloc[:, 1]
# ratios_df['ratio2'] = distances_df.iloc[:, 0] / distances_df.iloc[:, 2]
# ratios_df['ratio3'] = distances_df.iloc[:, 0] / distances_df.iloc[:, 3]
# print(distances_df.iloc[:, 0].values[0])
# # Calculate mean and std for each ratio
# ratios_summary = pd.DataFrame({
#     'mean_ratio1': [ratios_df['ratio1'].mean()],
#     'mean_ratio2': [ratios_df['ratio2'].mean()],
#     'mean_ratio3': [ratios_df['ratio3'].mean()],
#     'std_ratio1': [ratios_df['ratio1'].std()],
#     'std_ratio2': [ratios_df['ratio2'].std()],
#     'std_ratio3': [ratios_df['ratio3'].std()],
# })
# df_columns_baseline = ["Length", "Width  1", "width 2"]
# Jaw distances

df_columns_baseline = ["distance_1", "distance_2", "distance_3", "distance_4"]


# If you want to add mean and std as new columns in ratios_df
# for col in ratios_summary.columns:
#     ratios_df[col] = ratios_summary[col][0]

# print(ratios_df)


# df_merged = pd.concat([distances_df, ratios_df], axis=1)
# df_merged.to_csv('distances_ratio.csv', index=False)

# print(df_merged.head(5))


# def create_latex_table(data):
#     header = """\\begin{table}[ht]
# \\centering
# \\begin{tabular}{|c|c|c|c|}
# \\hline
# \\textbf{Mean Distance Index} & \\textbf{Lower Limit} & \\textbf{Upper Limit} & \\textbf{std}\\\\ \\hline
# """
#     footer = """\\end{tabular}
# \\caption{Descriptive Statistics of Mean Distances}
# \\label{tab:MeanDistances}
# \\end{table}
# """
#     body = ""
#     for d in data:
#         body += f"{d[0]} & {d[1]:.2f} & {d[2]:.2f} & {d[3]:.2f}\\\\ \\hline\n"

#     return header + body + footer


# # Number of bins
# n_bins = 20
# # Use a colormap
# colors = plt.cm.viridis(np.linspace(0, 1, n_bins))

# # Plotting the histogram for 'Length' column
# n, bins, patches = plt.hist(
#     df_merged['Length'], bins=n_bins, edgecolor='black', alpha=0.7)

# # Color each bin
# for i in range(n_bins):
#     patches[i].set_facecolor(colors[i])

# plt.title('Frequency Distribution of Length')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.show()

# df_width_length = pd.DataFrame(
#     data, columns=['Length', 'widht 1 ', 'widht 2', 'widht 3'])

# df_individual = pd.DataFrame(
#     data, columns=['Length', 'widht 1 ', 'widht 2', 'widht 3'])
# # print(create_latex_table(data))
# # df_individual = pd.DataFrame(rows_individual, columns=df_columns_individual)
# df_individual.to_csv('df_individual.csv', index=False)
# length_450 = df_merged[df_merged['Length'] < 450]
# print(len(length_450)
# drop: Width  2,Width 3,ratio1,ratio2,ratio3,mean_ratio1,mean_ratio2,mean_ratio3,std_ratio1,std_ratio2,std_ratio3]


# column_drop = ['Width  1', 'Width 3', 'ratio1', 'ratio2', 'ratio3', 'mean_ratio1',
#                'mean_ratio2', 'mean_ratio3', 'std_ratio1', 'std_ratio2', 'std_ratio3']
# # save the dataframe as a csv file
# ratio_length = {
#     'ratio1': [],
#     'ratio2': [],
# }
# length_600 = df_merged[(df_merged['Length'] > 600) &
#                        (df_merged['Length'] < 900)]
# length_600 = length_600.drop(columns=column_drop)
# # length_450 = length_450.drop(columns=column_drop)
# for index, row in length_600.iterrows():
#     print(row[0])
#     print(row[1])
#     width_length_r = row[0] / row[1]
#     # length_450['ratio1'] = width_length_r
#     ratio_length['ratio1'].append(width_length_r)

# length_600['ratio1'] = ratio_length['ratio1']
# length_600['mean_ratio1'] = length_600['ratio1'].mean()
# length_600['std_ratio1'] = length_600['ratio1'].std()
# length_600.to_csv('length_900.csv', index=False)

# length_500 = df_merged[(df_merged['Length'] > 450) &
#
#                        (df_merged['Length'] < 500)]

# length_600.to_csv('length_600.csv', index=False)


# # Assuming you already have length_450 and length_500 dataframes as mentioned in your code
# length_550['Length_Category'] = '< 450'
# length_550['Length_Category'] = '450 - 500'
# combined_df = pd.concat([length_550, length_500])

# # Plot
# sns.jointplot(data=combined_df, x='Length', y='Width  2',
#               hue='Length_Category', height=8)
# plt.suptitle('Joint Distribution of Width 2 vs Length', y=1.02)

# plt.show()

# # 2. Violin Plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(data=combined_df, x='Length_Category', y='Width  2')
# plt.title('Distribution of Width 2 for each Length Category')

# plt.show()

# Find the ratio of length and width


# df = pd.DataFrame(distances_df)
# df.columns = ['Length', 'Width  1', 'width 2']
# for index, row in df.iterrows():
#     print(row[0])
#     print(row[1])
#     width_length_r = row[0] / row[1]
#     # length_450['ratio1'] = width_length_r
#     width_length_r_2 = row[0] / row[2]
#     ratio_length['ratio1'].append(width_length_r)
#     ratio_length['ratio2'].append(width_length_r_2)

# df['ratio1'] = ratio_length['ratio1']
# df['mean'] = df['ratio1'].mean()
# df['stand'] = df['ratio1'].std()
# df['ratio2'] = ratio_length['ratio2']
# df['mean2'] = df['ratio2'].mean()
# df['stand2'] = df['ratio2'].std()
# df.to_csv('thin_thickness_data.csv', index=False)
# print("Done")

"""
This the jaw distance using 0-2, and 1-3"""
column_means = distances_df.mean()
column_std_devs = distances_df.std()
print(column_means)
print(column_std_devs)

new_table = pd.DataFrame({
    'mean': column_means,
    'std': column_std_devs,
    'lower_limit': column_means - 2 * column_std_devs,
    'upper_limit': column_means + 2 * column_std_devs,
})
print(new_table)
# SAVE THE DATAFRAME AS A CSV FILE

new_table.to_csv('spine_distances.csv', index=False)


# for index, row in distances_df.iterrows():
#     print(row[0])
#     print(row[1])
# width_length_r = row[0] / row[1]
# # length_450['ratio1'] = width_length_r
# width_length_r_2 = row[0] / row[2]
# ratio_length['ratio1'].append(width_length_r)
# ratio_length['ratio2'].append(width_length_r_2)
