import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from math import sqrt
import sys

#.................Function to Euclidean distance.............................

def euclidean_distance(data_point,represenantive):
   
    
    dist = np.linalg.norm(data_point-represenantive)
    return dist


#.................Function to Manhatten distance.............................
def manhatten_distance(data_point,represenantive):
    if isinstance(data_point, float) :
        return abs((data_point-represenantive))

    else:
        dimension = len(data_point)      
        s = 0
        for dim in range(0, dimension):
            s+= abs(data_point[dim] - represenantive[dim])
   
    return sqrt(s)   
    

#.................Function to compute error_bw_centroid.............................    
    
def error_bw_centroid(old_represenantive,new_represenantive):
    
    length=old_represenantive.shape[0]
    error=0
    for i in range(length):
        error+=np.sum((old_represenantive[i]-new_represenantive[i])**2)
    return error    
        
#...................  Function to compute convergence...................  
def convergence(old_represenantive,new_represenantive,iteration,max_iterations,max_error):
    error= error_bw_centroid(np.array(old_represenantive),np.array(new_represenantive))
    #print("error=",e)
    
    if iteration > max_iterations or error < max_error :
        
        return True
    return False
    
#.................. Function for reassign the points in clusters...............   
def reassignment_of_points(data,old_represenantive,clustering_algo): 
    final_cluster_list={}
 
    for i in range(1,no_of_clusters+1):
             final_cluster_list.setdefault(i,[])
             
    if clustering_algo == "kmeans":       
             
        for data_point in data:
            distance_list=[] 
            
            for represenantive in old_represenantive:
                    distance_list.append(euclidean_distance(data_point,represenantive))
                    
            #...store the distance of a point to all clusters.......
            distance_list=np.array(distance_list) 
                   
            #find index of minimum distance
            index_of_cluster_to_assign=np.argmin(distance_list)+1 
            #assign point to that cluster wich is nearer to it
            final_cluster_list[index_of_cluster_to_assign].append(data_point)
        return final_cluster_list
        
   
    
#.................. Function for recomputing/updating represenative in clusters...............     
def recompute_representative(final_cluster_list,clustering_algo):
    new_representative=[]
    if clustering_algo == "kmeans":
        for cluster_no,points_in_clusters in final_cluster_list.items():
              
                m=np.mean(np.array(points_in_clusters),axis=0)
                new_representative.append(m)  
        return new_representative
   
  
#...................Function for kmeans.................  
          
def kmeans(data,no_of_clusters,representative_list):
    max_iterations=100 
    max_error=0.0001
    iteration=0    
    
    old_representative=representative_list.copy()
    print("\nInitial Represenatatives=",representative_list)
    #for each of the datapoint
    while 1: 
        new_representative=[]
        final_cluster_list={}
        iteration+=1
       
        
        final_cluster_list=reassignment_of_points(data,old_representative,"kmeans")
        #print(final_cluster_list)
            
       #Recomputation of Representative points
        new_representative=recompute_representative(final_cluster_list,"kmeans")   
        print('\nAfter iteration {}..' .format(iteration))
        print('New Represenatatives are:',new_representative) 
#...................printing the clusters and their count...........
        for cluster_no,values in final_cluster_list.items():
                points=[]
                for i in values:
                    points.append(i.tolist())
                print("\nNo of Data points in cluster {}={}\n".format(cluster_no,len(points)) )
                print("Data points are: {}".format(points))
#.............................................................................           
          #....check if algo converges............................
        if convergence(old_representative,new_representative,iteration,max_iterations,max_error) is True:
           break;
           
            
        old_representative = new_representative.copy()  #copy new centroid to old centroid list
        
     
    print('\nK-means converged after {} iterations' .format(iteration))
    return final_cluster_list ,new_representative   



            
if __name__ == '__main__':
    print("choices")
    no_of_clusters=3
    print("1.K means")
    choice=int(input("Enter choice"))
    if choice == 1:
        print("1.Generate Random Represenative ")
        ch=int(input("Enter choice"))      
        if ch == 1:
           
            #.....load the dataset..............
            data=np.loadtxt(sys.argv[1],delimiter=',')       
          
          
           
            #.........Generate random centroid.......................
            repr_index=[]
            represenatative_list=[]
            for i in range(1,no_of_clusters+1):
                repr_index.append(random.randint(0,data.shape[0]-1))
            while len(repr_index)!=len(set(repr_index)):
                repr_index=[]
                for i in range(1,no_of_clusters+1):
                    repr_index.append(random.randint(0,data.shape[0]-1))
                    
            for i in repr_index:
                represenatative_list.append(data[i]) 
            #............................................................
        
        #.......printing the Final Representative points
        final_cluster_list,new_represenatative=kmeans(data,no_of_clusters,represenatative_list)
        print("\nFinal Representative points")
        for point in new_represenatative:
         print(point)
          
         #..printing the final clusters and their count........... 
        
        j=1
        print("Final clusters")  
        for cluster_no,values in final_cluster_list.items():
            points=[]
            for i in values:
                points.append(i.tolist())
            filename='benign_points'+str(j)+'.csv'
            np.savetxt(filename, np.array(values)) 
            j=j+1
            print("\nNo of Data points in cluster {}={}\n".format(cluster_no,len(points)) )
           # print("Data points are: {}".format(points))
        np.savetxt('benign_centroid_points.csv',np.array(new_represenatative))
        

#        we=np.loadtxt('malignant_centroid_points.csv',delimiter=',') 
#        print(we[0])
   

 #............................End of program....................................       
        
        
        
        
      
 
    




