#This script applies the keppler mapper algorithm on a standardized dataset (i.e. a 110x7 matrix)

from utils import *
import numpy as np
from sklearn.metrics import silhouette_score as ss
import matplotlib.pyplot as plt
import kmapper as km
import sklearn
import sklearn.manifold as manifold

relative_difference_mean_max_minus_0_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_0_deg_6_corrected_measure_r.csv")
relative_difference_mean_max_minus_0_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_0_deg_6_corrected_measure_l.csv")

relative_difference_mean_max_minus_3_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_3_deg_9_corrected_measure_r.csv")
relative_difference_mean_max_minus_3_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_3_deg_9_corrected_measure_l.csv")


relative_difference_mean_max_minus_5_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_5_deg_4_corrected_measure_r.csv")
relative_difference_mean_max_minus_5_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_5_deg_4_corrected_measure_l.csv")


relative_difference_mean_max_minus_7_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_7_deg_6_corrected_measure_r.csv")
relative_difference_mean_max_minus_7_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_7_deg_6_corrected_measure_l.csv")


relative_difference_mean_max_minus_9_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_9_deg_4_corrected_measure_r.csv")
relative_difference_mean_max_minus_9_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_9_deg_4_corrected_measure_l.csv")

relative_difference_mean_max_minus_11_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_11_deg_5_corrected_measure_r.csv")
relative_difference_mean_max_minus_11_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_11_deg_5_corrected_measure_l.csv")

relative_difference_mean_max_minus_13_deg_r = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_13_deg_4_corrected_measure_r.csv")
relative_difference_mean_max_minus_13_deg_l = pd.read_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization3\\relative_difference_Right_hip_red_max_minus_13_deg_4_corrected_measure_l.csv")

#this function concatenates the 2 dataframes containing respectively the 55 parameters for the right side
# and the 55 parameters for the left side
def matrix(relative_difference_r, relative_difference_l):
    relative_difference_r_values = relative_difference_r.iloc[:]['0']
    relative_difference_l_values = relative_difference_l.iloc[:]['0']
    relative_difference_r_df = pd.DataFrame(relative_difference_r_values)
    relative_difference_l_df = pd.DataFrame(relative_difference_l_values)
    both_left_and_right_concatenated = pd.DataFrame()
    both_left_and_right_concatenated = pd.concat([both_left_and_right_concatenated, relative_difference_r_df],
                                                 axis=0)
    # both_left_and_right_concatenated is a 110x1 matrix
    both_left_and_right_concatenated = pd.concat([both_left_and_right_concatenated, relative_difference_l_df],
                                                 axis=0, ignore_index=True)
    return both_left_and_right_concatenated

both_left_and_right_concatenated_max_minus_0_deg = matrix(relative_difference_mean_max_minus_0_deg_r, relative_difference_mean_max_minus_0_deg_l)

both_left_and_right_concatenated_max_minus_3_deg = matrix(relative_difference_mean_max_minus_3_deg_r, relative_difference_mean_max_minus_3_deg_l)

both_left_and_right_concatenated_max_minus_5_deg = matrix(relative_difference_mean_max_minus_5_deg_r, relative_difference_mean_max_minus_5_deg_l)

both_left_and_right_concatenated_max_minus_7_deg = matrix(relative_difference_mean_max_minus_7_deg_r, relative_difference_mean_max_minus_7_deg_l)

both_left_and_right_concatenated_max_minus_9_deg = matrix(relative_difference_mean_max_minus_9_deg_r, relative_difference_mean_max_minus_9_deg_l)

both_left_and_right_concatenated_max_minus_11_deg = matrix(relative_difference_mean_max_minus_11_deg_r, relative_difference_mean_max_minus_11_deg_l)

both_left_and_right_concatenated_max_minus_13_deg = matrix(relative_difference_mean_max_minus_13_deg_r, relative_difference_mean_max_minus_13_deg_l)

both_left_and_right_concatenated_all = pd.DataFrame()
both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_0_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_3_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_5_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_7_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_9_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_11_deg],
                                                 axis=1)

both_left_and_right_concatenated_all = pd.concat([both_left_and_right_concatenated_all, both_left_and_right_concatenated_max_minus_13_deg],
                                                 axis=1)
both_left_and_right_concatenated_names = both_left_and_right_concatenated_max_minus_0_deg.copy()
#both_left_and_right_concatenated_names["name"] = {0: "knee_angle_mean_stance_r", 1: "knee_angle_std_stance_r",
                                                  #2:"knee_angle_min_stance_r", 3:"knee_angle_max_stance_r" ,
                                                  #4:"knee_angle_mean_swing_r", 5: "knee_angle_std_swing_r",
                                                  #6:"knee_angle_min_swing_r", 7: "knee_angle_max_swing_r",
                                                  #8:"ankle_angle_mean_stance_r", 9:"ankle_angle_std_stance_r",
                                                  #10:"ankle_angle_min_stance_r", 11:"ankle_angle_max_stance_r",
                                                  #12:"ankle_angle_mean_swing_r", 13:"ankle_angle_std_swing_r",
                                                  #14:"ankle_angle_min_swing_r", 15:"ankle_angle_max_swing_r",
                                                  #16:"hip_angle_mean_stance_r", 17:"hip_angle_std_stance_r",
                                                  #18:"hip_angle_min_stance_r", 19:"hip_angle_max_stance_r",
                                                  #20:"hip_angle_mean_swing_r", 21:"hip_angle_std_swing_r",
                                                  #22:"hip_angle_min_swing_r", 23:"hip_angle_max_swing_r",
                                                  #24:"knee_moment_mean_stance_r", 25:"knee_moment_std_stance_r",
                                                  #26:"knee_moment_min_stance_r", 27: "knee_moment_max_stance_r",
                                                  #28:"knee_moment_mean_swing_r", 29:"knee_moment_std_swing_r",
                                                  #30:"knee_moment_min_swing_r", 31:"knee_moment_max_swing_r",
                                                  #32:"ankle_moment_mean_stance_r", 33:"ankle_moment_std_stance_r",
                                                  #34:"ankle_moment_min_stance_r", 35:"ankle_moment_max_stance_r",
                                                  #36:"ankle_moment_mean_swing_r", 37:"ankle_moment_std_swing_r",
                                                  #38:"ankle_moment_min_swing_r", 39:"ankle_moment_max_swing_r",
                                                  #40:"hip_moment_mean_stance_r", 41:"hip_moment_std_stance_r",
                                                  #42:"hip_moment_min_stance_r", 43:"hip_moment_max_stance_r",
                                                  #44:"hip_moment_mean_swing_r", 45:"hip_moment_std_swing_r",
                                                  #46:"hip_moment_min_swing_r", 47:"hip_moment_max_swing_r",
                                                  #48:"grf_mean_stance_r", 49:"grf_std_stance_r",
                                                  #50:"grf_min_stance_r", 51:"grf_max_stance_r",
                                                  #52:"grf_mean_swing_r", 53:"grf_std_swing_r",
                                                  #54:"grf_max_swing_r", 55:"knee_angle_mean_stance_l",
                                                  #56:"knee_angle_std_stance_l", 57:"knee_angle_min_stance_l",
                                                  #58:"knee_angle_max_stance_l", 59:"knee_angle_mean_swing_l",
                                                  #60:"knee_angle_std_swing_l", 61:"knee_angle_min_swing_l",
                                                  #62:"knee_angle_max_swing_l", 63:"ankle_angle_mean_stance_l",
                                                  #64:"ankle_angle_std_stance_l", 65:"ankle_angle_min_stance_l",
                                                  #66:"ankle_angle_max_stance_l", 67:"ankle_angle_mean_swing_l",
                                                  #68:"ankle_angle_std_swing_l", 69:"ankle_angle_min_swing_l",
                                                  #70:"ankle_angle_max_swing_l", 71:"hip_angle_mean_stance_l",
                                                  #72:"hip_angle_std_stance_l", 73:"hip_angle_min_stance_l",
                                                  #74:"hip_angle_max_stance_l", 75:"hip_angle_mean_swing_l",
                                                  #76:"hip_angle_std_swing_l", 77:"hip_angle_min_swing_l",
                                                  #78:"hip_angle_max_swing_l", 79:"knee_moment_mean_stance_l",
                                                  #80:"knee_moment_std_stance_l", 81:"knee_moment_min_stance_l",
                                                  #82:"knee_moment_max_stance_l", 83:"knee_moment_mean_swing_l",
                                                  #84:"knee_moment_std_swing_l", 85:"knee_moment_min_swing_l",
                                                  #86:"knee_moment_max_swing_l", 87:"ankle_moment_mean_stance_l",
                                                  #88:"ankle_moment_std_stance_l", 89:"ankle_moment_min_stance_l",
                                                  #90:"ankle_moment_max_stance_l", 91:"ankle_moment_mean_swing_l",
                                                  #92:"ankle_moment_std_swing_l", 93:"ankle_moment_min_swing_l",
                                                  #94:"ankle_moment_max_swing_l", 95:"hip_moment_mean_stance_l",
                                                  #96:"hip_moment_std_stance_l", 97:"hip_moment_min_stance_l",
                                                  #98:"hip_moment_max_stance_l", 99:"hip_moment_mean_swing_l",
                                                  #100:"hip_moment_std_swing_l", 101:"hip_moment_min_swing_l",
                                                  #102:"hip_moment_max_swing_l", 103:"grf_mean_stance_l",
                                                  #104:"grf_std_stance_l", 105:"grf_min_stance_l",
                                                  #106:"grf_max_stance_l", 107:"grf_mean_swing_l",
                                                  #108:"grf_std_swing_l",109:"grf_max_swing_l"}



both_left_and_right_concatenated_names["name"] = {0: "knee_angle_mean_stance_r", 1: "knee_angle_std_stance_r",
                                                  2:"knee_angle_min_stance_r", 3:"knee_angle_max_stance_r" ,
                                                  4:"knee_angle_mean_swing_r", 5: "knee_angle_std_swing_r",
                                                  6:"knee_angle_min_swing_r", 7: "knee_angle_max_swing_r",
                                                  8:"ankle_angle_mean_stance_r", 9:"ankle_angle_std_stance_r",
                                                  10:"ankle_angle_min_stance_r", 11:"ankle_angle_max_stance_r",
                                                  12:"ankle_angle_mean_swing_r", 13:"ankle_angle_std_swing_r",
                                                  14:"ankle_angle_min_swing_r", 15:"ankle_angle_max_swing_r",
                                                  16:"hip_angle_mean_stance_r", 17:"hip_angle_std_stance_r",
                                                  18:"hip_angle_min_stance_r", 19:"hip_angle_max_stance_r",
                                                  20:"hip_angle_mean_swing_r", 21:"hip_angle_std_swing_r",
                                                  22:"hip_angle_min_swing_r", 23:"hip_angle_max_swing_r",
                                                  24:"knee_moment_mean_stance_r", 25:"knee_moment_std_stance_r",
                                                  26:"knee_moment_min_stance_r", 27: "knee_moment_max_stance_r",
                                                  28:"knee_moment_mean_swing_r", 29:"knee_moment_std_swing_r",
                                                  30:"knee_moment_min_swing_r", 31:"knee_moment_max_swing_r",
                                                  32:"ankle_moment_mean_stance_r", 33:"ankle_moment_std_stance_r",
                                                  34:"ankle_moment_min_stance_r", 35:"ankle_moment_max_stance_r",
                                                  36:"hip_moment_mean_stance_r", 37:"hip_moment_std_stance_r",
                                                  38:"hip_moment_min_stance_r", 39:"hip_moment_max_stance_r",
                                                  40:"hip_moment_mean_swing_r", 41:"hip_moment_std_swing_r",
                                                  42:"hip_moment_min_swing_r", 43:"hip_moment_max_swing_r",
                                                  44:"grf_mean_stance_r", 45:"grf_std_stance_r",
                                                  46:"grf_min_stance_r", 47:"grf_max_stance_r",
                                                  48:"pelvis_tilt_mean_stance_r", 49:"pelvis_tilt_std_stance_r",
                                                  50:"pelvis_tilt_min_stance_r", 51:"pelvis_tilt_max_stance_r",
                                                  52: "pelvis_tilt_mean_swing_r", 53: "pelvis_tilt_std_swing_r",
                                                  54: "pelvis_tilt_min_swing_r", 55: "pelvis_tilt_max_swing_r",
                                                  56:"knee_angle_mean_stance_l",
                                                  57:"knee_angle_std_stance_l", 58:"knee_angle_min_stance_l",
                                                  59:"knee_angle_max_stance_l", 60:"knee_angle_mean_swing_l",
                                                  61:"knee_angle_std_swing_l", 62:"knee_angle_min_swing_l",
                                                  63:"knee_angle_max_swing_l", 64:"ankle_angle_mean_stance_l",
                                                  65:"ankle_angle_std_stance_l", 66:"ankle_angle_min_stance_l",
                                                  67:"ankle_angle_max_stance_l", 68:"ankle_angle_mean_swing_l",
                                                  69:"ankle_angle_std_swing_l", 70:"ankle_angle_min_swing_l",
                                                  71:"ankle_angle_max_swing_l", 72:"hip_angle_mean_stance_l",
                                                  73:"hip_angle_std_stance_l", 74:"hip_angle_min_stance_l",
                                                  75:"hip_angle_max_stance_l", 76:"hip_angle_mean_swing_l",
                                                  77:"hip_angle_std_swing_l", 78:"hip_angle_min_swing_l",
                                                  79:"hip_angle_max_swing_l", 80:"knee_moment_mean_stance_l",
                                                  81:"knee_moment_std_stance_l", 82:"knee_moment_min_stance_l",
                                                  83:"knee_moment_max_stance_l", 84:"knee_moment_mean_swing_l",
                                                  85:"knee_moment_std_swing_l", 86:"knee_moment_min_swing_l",
                                                  87:"knee_moment_max_swing_l",
                                                  88:"ankle_moment_mean_stance_l",89:"ankle_moment_std_stance_l",
                                                  90:"ankle_moment_min_stance_l",91:"ankle_moment_max_stance_l",
                                                  92:"hip_moment_mean_stance_l",
                                                  93:"hip_moment_std_stance_l", 94:"hip_moment_min_stance_l",
                                                  95:"hip_moment_max_stance_l", 96:"hip_moment_mean_swing_l",
                                                  97:"hip_moment_std_swing_l", 98:"hip_moment_min_swing_l",
                                                  99:"hip_moment_max_swing_l", 100:"grf_mean_stance_l",
                                                  101:"grf_std_stance_l", 102:"grf_min_stance_l",
                                                  103:"grf_max_stance_l",
                                                  104: "pelvis_tilt_mean_stance_l", 105: "pelvis_tilt_std_stance_l",
                                                  106: "pelvis_tilt_min_stance_l", 107: "pelvis_tilt_max_stance_l",
                                                  108: "pelvis_tilt_mean_swing_l", 109: "pelvis_tilt_std_swing_l",
                                                  110: "pelvis_tilt_min_swing_l", 111: "pelvis_tilt_max_swing_l" }




#"both_left_and_right_concatenated_all_numpy" is a 110x7 matrix, it is our dataset
both_left_and_right_concatenated_all_numpy = both_left_and_right_concatenated_all.to_numpy()
both_left_and_right_concatenated_all_numpy_transpose = np.transpose(both_left_and_right_concatenated_all_numpy)
both_left_and_right_concatenated_all_numpy_transpose= np.abs(both_left_and_right_concatenated_all_numpy_transpose)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model = scaler.fit(both_left_and_right_concatenated_all_numpy_transpose)
scaled_data = model.transform(both_left_and_right_concatenated_all_numpy_transpose)
scaled_data_transpose = np.transpose(scaled_data)



#The Nearest Neighbors plot facilitates the identification of the range of epsilon values to be explored
# in the Kepler Mapper algorithm.
from sklearn.neighbors import NearestNeighbors
# Compute nearest neighbor distances
k = 3
nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data_transpose)
distances_output, indices = nbrs.kneighbors(scaled_data_transpose)
distances = np.sort(distances_output, axis = 0)
distances = distances[:,1]


# Plot the distances
plt.plot(range(0, len(both_left_and_right_concatenated_all)), distances, marker='o')
plt.xlabel('Data Points')
plt.ylabel('Nearest Neighbor Distance' )
plt.title('Nearest Neighbor Distances Plot k =' + str(k))
plt.show()

#eps = [0.5; 1] if we use z_score witout abs
epsilons = np.linspace(0.2, 0.5, num = 7)
min_samples = np.arange(2,20,step = 3)
n_cubes = np.arange(8,15,step = 1)
perc_overlap = np.linspace(0.1, 0.5, num = 5)
neighbohrs = np.arange(5,20,step = 3)

import itertools
mapper = km.KeplerMapper(verbose=1)


combinations = list(itertools.product(epsilons, min_samples, n_cubes, perc_overlap, neighbohrs))
print(combinations)

#This function tries to find the set of parameters that lead to the highest silhouette score.
def get_scores_and_labels(combinations, X):
    scores =[]
    all_labels_list = []

    for i, (eps, num_samples, n_cubes, perc_overlap, neigh) in enumerate(combinations):

        projected_data = mapper.fit_transform(X, projection =manifold.Isomap(n_components=2, n_neighbors=neigh, n_jobs=-1))
        G = mapper.map(projected_data,X, clusterer=sklearn.cluster.DBSCAN(eps=eps, min_samples=num_samples), cover=km.Cover(n_cubes = n_cubes, perc_overlap=perc_overlap))

        cluster_name = list(G["nodes"].keys())
        label = np.zeros(len(X))

        label_df = pd.DataFrame({'label': label})

        X = pd.DataFrame(X)
        X = pd.concat([X, label_df], axis=1)
        X = X.astype(float)

        for k in range(len(cluster_name)):
            cluster_members = G["nodes"][cluster_name[k]]

            array_with_name = np.zeros(len(cluster_members))
            array_with_name = array_with_name.astype(str)
            for j in range(len(cluster_members)):
                id = cluster_members[j]
                X.loc[id, 'label'] = k
                array_with_name[j] = both_left_and_right_concatenated_names.iloc[id]['name']

        all_labels = X.iloc[:]['label'].values
        X = X.drop('label', axis=1)


        labels_set = set(all_labels)
        num_cluster = len(labels_set)


        if (num_cluster < 2) or (num_cluster > 50):
            scores.append(-10)
            all_labels_list.append('bad')
            continue

        scores.append(ss(X, all_labels))
        all_labels_list.append(all_labels)
        c = (eps, num_samples, n_cubes, perc_overlap, neigh)
        print(f"Index {i}, eps:{eps}, min_sample: {num_samples}, perc_overlap: {perc_overlap}, n_cubes : {n_cubes}, cluster_name: {cluster_name}, Score : {scores[-1]}, neighbohrs ISOMAP : {neigh}")

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]


    return {'best_epsilon': best_parameters[0],
            'best_min_samples' : best_parameters[1],
            'best_n_cubes': best_parameters[2],
            'best_perc_overlap': best_parameters[3],
            'best_labels' : best_labels,
            'best_score' : best_score,
            'best_neighbohrs_ISOMAP' : best_parameters[4]}


best_dict2 = get_scores_and_labels(combinations, scaled_data_transpose)

print(best_dict2)
best_eps = best_dict2['best_epsilon']
best_min_sample = best_dict2['best_min_samples']
best_n_cubes = best_dict2['best_n_cubes']
best_perc_overlap = best_dict2['best_perc_overlap']
best_neighbohrs_ISOMAP = best_dict2['best_neighbohrs_ISOMAP']


#Compute the Keppler mapper using the set of parameters that led to the highest silhouette score
mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.fit_transform(scaled_data_transpose, projection =manifold.Isomap(n_components=2, n_neighbors=best_neighbohrs_ISOMAP, n_jobs=-1))

G = mapper.map(projected_data, scaled_data_transpose, clusterer=sklearn.cluster.DBSCAN(eps=best_eps
, min_samples=best_min_sample), cover=km.Cover(n_cubes = best_n_cubes, perc_overlap=best_perc_overlap))

mapper.visualize(G, path_html="mapper_deviations_both_sides_best_values_" + "decrease_hip_extension" + ".html")


########################################################################################################


cluster_name = list(G["nodes"].keys())
label = np.zeros(len(both_left_and_right_concatenated_all))

label_df = pd.DataFrame({'label':label})

both_left_and_right_concatenated = pd.concat([both_left_and_right_concatenated_all, label_df], axis = 1)
both_left_and_right_concatenated = both_left_and_right_concatenated.astype(float)

#Print the parameters that are within each node/cluster within the the Keppler map
for i in range(len(cluster_name)):
    cluster_members = G["nodes"][cluster_name[i]]

    array_with_name = np.zeros(len(cluster_members))
    array_with_name = array_with_name.astype(str)
    for j in range(len(cluster_members)):
        id = cluster_members[j]
        both_left_and_right_concatenated.loc[id, 'label'] = i
        array_with_name[j] = both_left_and_right_concatenated_names.iloc[id]['name']

    print(cluster_name[i])
    print(array_with_name)
