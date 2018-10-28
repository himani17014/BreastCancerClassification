import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import scipy.spatial
from scipy.spatial.distance import cdist
import math


k=3  # No of clusters

dist_fn = 'euclidean'
type = 'kmeans'

# dist_fn = 'manhattan'
# type = 'kmedian'

mal_data = np.loadtxt("data/malignant.csv",float,delimiter=',',skiprows=1)
ben_data = np.loadtxt("data/benign.csv",float,delimiter=',',skiprows=1)
# lable = np.loadtxt("data.csv",str,delimiter=',',skiprows=1,usecols=(1))

mal_label = np.ndarray((len(mal_data),1))
ben_label = np.ndarray((len(ben_data),1))

mal_label.fill(1)
ben_label.fill(0)


data = mal_data
lable = mal_label

# data = ben_data
# lable = ben_label

lable_int = np.ndarray((len(lable),1))

for i in range(len(lable)):
    if lable[i,0] == 'M':
        lable_int[i,0] = 0
    else:
        lable_int[i,0] = 1
# faulty case
# seeds = np.array([[0.69900126, 0.59815354],
#                   [0.97918822, 0.07715065],
#                   [0.69900126, 0.59815354]])

seeds = data[np.random.randint(0, len(data) - 1, size=k)]
print seeds.shape
print seeds

# Plot 2D data
# plt.scatter(data[:,0], data[:,1])
# plt.savefig("plot2D")


def distance_fn(a,b,distance):
    if distance == 'euclidean':
        return np.nan_to_num(np.linalg.norm(a-b))
    elif distance == 'manhattan':
        return np.nan_to_num(scipy.spatial.distance.cityblock(a,b))

def find_new_centroids(near_pts,data,type):
    if type == 'kmeans':
        new_centroid = np.nan_to_num(np.mean(data[near_pts], axis=0))  # for kmeans; calculate median for k medians
        return new_centroid
    elif type == 'kmedian':
        new_centroid = np.nan_to_num(np.median(data[near_pts], axis=0))  # for kmeadian;
        return new_centroid

def clusterk(k,centroids,data,distance='euclidean',type='kmeans'):

    nrows, ncol = data.shape
    print "Data Shape",data.shape

    belongs_to = np.zeros((nrows, 1),dtype=int)
    # print centroids
    # print "centroid shape",centroids.shape
    old_centroids = np.zeros(centroids.shape) #centroids is a list
    curr_centroids = centroids

    it=0
    # dist = distance_fn(curr_centroids,old_centroids,distance)
    # print dist

    while not np.array_equal(old_centroids,curr_centroids):
    # while dist>0:
    # while it<5:
    #     dist = distance_fn(curr_centroids,old_centroids,distance)
        it += 1
        print it
        # for each datapoint
        for ind, val in enumerate(data):
            dist_vector = np.zeros((k,1))
            for ind_centroid, val_centroid in enumerate(curr_centroids):
                dist_cent_point = distance_fn(val_centroid,val,distance)
                # if not math.isnan(dist_cent_point):
                dist_vector[ind_centroid] = dist_cent_point
            # print "dist vector:\n", dist_vector
            belongs_to[ind,0] = np.argmin(dist_vector)
            # print belongs_to

        centroid_tmp = np.zeros((k,ncol))

        # for each cluster
        for index in range(len(curr_centroids)):
            #get all pts assigned to cluster
            near_pts = []
            for i in range(len(belongs_to)):
                if belongs_to[i] == index:
                    near_pts.append(i)

            # Find new centroids (kmeans / kmedian)
            new_centroid = find_new_centroids(near_pts,data,type)
            centroid_tmp[index, :] = new_centroid

        old_centroids = curr_centroids
        curr_centroids = centroid_tmp

        # print curr_centroids

    return curr_centroids,belongs_to,it


def plot1d(centroids,belongs_to,data,type,k,seed_name):
    plt.scatter(data,len(data)*[0],c=belongs_to)
    plt.scatter(centroids,len(centroids)*[0],c=['r','r','r'])
    plt.savefig("1D_" + str(type) + seed_name+"_plot_k"+str(k))


def plot2d(centroids,belongs_to,data,type,k):
    plt.scatter(data[:,0], data[:,1],c=belongs_to)
    plt.scatter(centroids[:,0], centroids[:,1], c=['r', 'r', 'r'])
    plt.savefig("2D_" + str(type) + "_plot_k"+str(k))


# For Q1, 2 ,3
centroids, belongs_to,it = clusterk(k,seeds,data,dist_fn,type)
print "Centroids",centroids
print "belongs to",belongs_to
print it

np.savetxt("centroids_malignant.csv",centroids,delimiter=',')


colors = ['blue', 'green', 'magenta', 'cyan', 'pink', 'orange', 'red', 'grey', 'aquamarine', 'lime']

belongs_to = belongs_to.reshape((len(belongs_to),))

# Plot 3d using TSNE
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# tsne = TSNE(n_components = 3, init = 'pca', angle = 0.9)
# X_tsne = tsne.fit_transform(data)
# print len(X_tsne)
# plt.figure(figsize = (12, 8))
# for i in range(0, len(X_tsne[:,0])):
#     for j in range(0,6):
#         if lable_int[i,0] == j:
#             ax.scatter( X_tsne[i][1],X_tsne[i][0],X_tsne[i][2], marker='x', c = colors[belongs_to[i]], s = 30, edgecolors = 'face')
#             break
#
# plt.savefig('Fig2.jpg')
# plt.show()


# Plot 2d using TSNE
tsne = TSNE(n_components = 2, init = 'pca', angle = 0.9)
X_tsne = tsne.fit_transform(data)
print len(X_tsne)
plt.figure(figsize = (12, 8))
for i in range(0, len(X_tsne[:,0])):
    for j in range(0,6):
        if lable_int[i,0] == j:
            plt.scatter( X_tsne[i][0],X_tsne[i][1], marker='x', c = colors[belongs_to[i]], s = 30, edgecolors = 'face')
            break

plt.savefig('Fig.jpg')
plt.show()


# plot2d(centroids,belongs_to,data,type,k)



