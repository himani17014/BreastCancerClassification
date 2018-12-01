import numpy as np
import math

n_cluster = 6
n_feature = 30
n_samples = 569

# data = np.loadtxt("data/data.csv",float,delimiter=',',skiprows=1,usecols=range(2,32))
# lable = np.loadtxt("data/data.csv",str,delimiter=',',skiprows=1,usecols=(1))
# lable = lable.reshape((len(lable),1))
#
# malignant = np.empty((0,data.shape[1]))
# benign =  np.empty((0,data.shape[1]))
#
# print data[0].shape
# print lable.shape
# # print lable
#
# for i in range(len(data)):
#     if lable[i] == 'M':
#         malignant = np.append(malignant,[data[i]],axis=0)
#     else:
#         benign = np.append(benign,[data[i]],axis=0)
#
# print malignant.shape
# print benign.shape
#
# # fmt = ",".join(["%s"] + ["%10.6e"] * (malignant.shape[1]-1))
# head = "radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst,"
#
# np.savetxt("malignant.csv",malignant,delimiter=',',header=head)
# np.savetxt("benign.csv",benign,delimiter=',',header=head)


# ********************************************************************************************
centroids_ben = np.loadtxt("data/centroid_data/benign_centroid_points.csv",delimiter=' ',dtype=float)
centroids_mal = np.loadtxt("data/centroid_data/malignant_centroid_points.csv",delimiter=' ',dtype=float)
# centroids_ben_lable = np.zeros
print centroids_ben.shape
print centroids_mal.shape

centroids = np.concatenate((centroids_ben,centroids_mal),axis=0)
# print centroids

c1_ben = np.loadtxt("data/centroid_data/benign_points1.csv",delimiter=' ',dtype=float)
c2_ben = np.loadtxt("data/centroid_data/benign_points2.csv",delimiter=' ',dtype=float)
c3_ben = np.loadtxt("data/centroid_data/benign_points3.csv",delimiter=' ',dtype=float)
c4_mal = np.loadtxt("data/centroid_data/malignant_points1.csv",delimiter=' ',dtype=float)
c5_mal = np.loadtxt("data/centroid_data/malignant_points2.csv",delimiter=' ',dtype=float)
c6_mal = np.loadtxt("data/centroid_data/malignant_points3.csv",delimiter=' ',dtype=float)

mal_data = np.loadtxt("data/malignant.csv",float,delimiter=',',skiprows=1)
ben_data = np.loadtxt("data/benign.csv",float,delimiter=',',skiprows=1)

mal_label = np.ndarray((len(mal_data),1))
ben_label = np.ndarray((len(ben_data),1))

mal_label.fill(1)
ben_label.fill(0)
lable = np.concatenate((ben_label,mal_label),axis=0)

print lable

list = []
list.append(c1_ben)
list.append(c2_ben)
list.append(c3_ben)
list.append(c4_mal)
list.append(c5_mal)
list.append(c6_mal)

print len(list)
# print list[0]


# arr = np.asarray(list)
# print arr.shape,c2_ben.shape
# arr = np.append(arr,[c2_ben],axis=2)
# print arr.shape
# np.savetxt("temp",arr)
min_feature_val = np.zeros((n_cluster,n_feature),dtype=float)
max_feature_val = np.zeros((n_cluster,n_feature),dtype=float)


# Reducing features from 30 to 6
for cluster_no in range(len(list)):
    for j in range(list[cluster_no].shape[1]):
        col = list[cluster_no][:,j]
        # print col
        # print np.min(col)
        # print np.abs(2-col)
        min_feature_val[cluster_no][j] = np.min(col)
        max_feature_val[cluster_no][j] = np.max(col)

print min_feature_val.shape
print max_feature_val.shape

Pmat = np.zeros((n_samples,n_cluster),dtype=float)

pt_no = -1
# to access points in each cluster
for cl in range(len(list)):
    for i in range(len(list[cl])):
        pt_no += 1
        for cluster_no in range(len(list)):
            f = 0.0
            for j in range(n_feature):
                if (list[cl][i,j] >= min_feature_val[cluster_no][j]) and \
                    (list[cl][i,j] <= max_feature_val[cluster_no][j]):
                    f += 1 - (np.abs(centroids[cluster_no][j] - list[cl][i,j]) /
                         float(np.max(np.abs(centroids[cluster_no][j]-list[cluster_no][:,j]))))
                else:
                    f += 0

            Pmat[pt_no][cluster_no] = f/float(n_feature)

print Pmat

np.savetxt("data/Reduced_feature_matrix.npy",Pmat,delimiter=',')
np.savetxt("data/lables",lable,delimiter=',')

final_Pmat = np.concatenate((Pmat,lable),axis=1)
np.random.shuffle(final_Pmat)
np.savetxt("data/final_pmat.csv",final_Pmat,delimiter=' ')

