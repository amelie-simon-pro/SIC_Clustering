import random
import numpy as np
from scipy.spatial import distance

class KMeansEuclidean:
    def __init__(self, nb_clusters, iterations=100):
        self.nb_clusters = nb_clusters
        self.iterations = iterations
        self.seed = 60
        
        print("init self.iterations", self.iterations)

    def fit(self, dataset, initialization_centroid, quantiles):
        print("euclidean")
        ###################
        # Initialization ##
        ###################
        
        if initialization_centroid == '++':
           print("initialization centroid", initialization_centroid)
           # of centroid from kmeans++ concept
           unique_centroids = []
           
           # Select the first centroid randomly
           first_centroid = np.array(random.choice(list(dataset)))
           unique_centroids.append(first_centroid)
           
           while len(unique_centroids) < self.nb_clusters:
               distances_init = []               
               # Calculate the distance of each point to the nearest centroid
               for point in dataset:
                   min_distance_init = float('inf')
                   for centroid in unique_centroids:
                       distance_init = np.linalg.norm(point - centroid)
                       if distance_init < min_distance_init:
                           min_distance_init = distance_init
                   distances_init.append(min_distance_init**2)              
               # Convert distances to probabilities
               total_distance_init = sum(distances_init)
               probabilities = [d / total_distance_init for d in distances_init]              
               # Select the next centroid with probability proportional to the distance
               next_centroid_index = np.random.choice(len(dataset), p=probabilities)
               next_centroid = dataset[next_centroid_index]               
               # Check for uniqueness (optional, since k-means++ inherently avoids duplicates)
               is_unique = True
               for centroid in unique_centroids:
                   if np.array_equal(next_centroid, centroid):
                       is_unique = False
                       break
               if is_unique:
                   unique_centroids.append(next_centroid)
                   
        elif initialization_centroid == 'quantile':
           print("initialization centroid", initialization_centroid)
           print("np.shape(quantiles)",np.shape(quantiles))
           unique_centroids=quantiles
               
        self.centroids = np.array(unique_centroids)
        print("np.shape(self.centroids)",np.shape(self.centroids))
        print("self.centroids",*self.centroids)
        
        ##############
        # Iteration ##
        ##############
        
        convergence_centroids = np.zeros((self.nb_clusters,self.iterations))
        for i in range(self.iterations):
            print("i euclidean", i)
            
            # Create an empty array to contain each cluster which will have different number of points
            partitions = np.empty(self.nb_clusters, dtype=object)
            for i_cluster in range(self.nb_clusters):
                partitions[i_cluster] = []
                
            self.labels = np.empty(len(dataset), dtype=int)
            # For each point, calculate distance to centroid, and assign to a cluster
            i_point=0
            for sample in dataset:
                distances = np.zeros(self.nb_clusters)
                for i_cluster in range(self.nb_clusters):
                   distances[i_cluster] = distance.euclidean(sample, self.centroids[i_cluster])
                self.labels[i_point] = np.argmin(distances)
                partitions[np.argmin(distances)].append(sample)
                i_point += 1
                
            # Keep the centroids before they're updated        
            old_centroids = np.copy(self.centroids)

            # Update the centroids
            for i_cluster in range(self.nb_clusters):
                self.centroids[i_cluster] = np.mean(partitions[i_cluster], axis=0)
                convergence_centroids[i_cluster,i]=np.sum(self.centroids[i_cluster] - old_centroids[i_cluster])
            
            #for i_cluster in range(self.nb_clusters):    
            #   print(f"np.shape(partition[{i_cluster}])",np.shape(partitions[i_cluster]))  
                    
            # Compare current centroid with the previous one  
            if np.allclose(self.centroids, old_centroids):
                break
 
        return np.array(self.labels), self.centroids, convergence_centroids

