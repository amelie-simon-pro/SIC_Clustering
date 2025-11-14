import random
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA

class KMeansMaha_cor:
    def __init__(self, nb_clusters, iterations=200):
        self.nb_clusters = nb_clusters
        self.iterations = iterations
        self.seed = 60

    def fit(self, dataset, initialization_centroid, quantiles):
        print("mahalanobis cor")
        ###################
        # Initialization ##
        ###################
              
        if initialization_centroid == '++':
           print("initialization centroid", initialization_centroid)
           unique_centroids = []
           
           # Select the first centroid randomly
           first_centroid = np.array(random.choice(list(dataset)))
           unique_centroids.append(first_centroid)
           
           while len(unique_centroids) < self.nb_clusters:
               distances_init = np.array([
                   min(np.linalg.norm(point - centroid) for centroid in unique_centroids)**2
                   for point in dataset
               ])
               
               # Convert distances to probabilities
               probabilities = distances_init / distances_init.sum()
               
               # Select the next centroid with probability proportional to the distance
               next_centroid_index = np.random.choice(len(dataset), p=probabilities)
               next_centroid = dataset[next_centroid_index]
               
               unique_centroids.append(next_centroid)

        elif initialization_centroid == 'quantile':
           print("initialization centroid", initialization_centroid)
           print("np.shape(quantiles)",np.shape(quantiles))
           unique_centroids=quantiles

        self.centroids = np.array(unique_centroids)
        print("initial self.centroid", self.centroids)
        print("np.shape(self.centroid)", np.shape(self.centroids))
        
        ##############
        # Iteration ##
        ##############
        Inv_Sigma = np.linalg.inv(np.corrcoef(dataset.T))
        print("np.shape(Inv_Sigma)", np.shape(Inv_Sigma))
        convergence_centroids = np.zeros((self.nb_clusters, self.iterations))
        
        for i in range(self.iterations):
            print("i mahalanobis cor ", i)
            
            distances = np.array([
                [distance.mahalanobis(sample, centroid, Inv_Sigma) for centroid in self.centroids]
                for sample in dataset
            ])
            
            self.labels = np.argmin(distances, axis=1)
            
            partitions = [dataset[self.labels == i] for i in range(self.nb_clusters)]
            
            old_centroids = np.copy(self.centroids)
            
            for i_cluster in range(self.nb_clusters):
                self.centroids[i_cluster] = np.mean(partitions[i_cluster], axis=0)
                convergence_centroids[i_cluster, i] = np.sum(self.centroids[i_cluster] - old_centroids[i_cluster])
            
            if np.allclose(self.centroids, old_centroids):
                break
 
        return self.labels, self.centroids, convergence_centroids
        
