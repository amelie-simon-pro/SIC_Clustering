import pandas as pd
import numpy as np
from scipy.spatial import distance

from netCDF4 import Dataset

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


from scipy.spatial import distance
from scipy.optimize import curve_fit

from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_hex, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
sns.set_style('whitegrid')

from datetime import datetime
START = datetime.now() 
import os

from kmeans_euclidean_v8 import KMeansEuclidean
from kmeans_maha_cor_v8 import KMeansMaha_cor

##################
### SUBROUTINE ###
##################

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
def create_color_range(start_color, end_color, num_colors=150):
    # Convert hex color codes to RGB
    start_rgb = np.array([int(start_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    #print("start_rgb",start_rgb)
    end_rgb = np.array([int(end_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

    # Create a color range
    color_range = [start_rgb + (end_rgb - start_rgb) * i / (num_colors - 1) for i in range(num_colors)]
    #print("color_range",color_range)

    # Create a colormap
    cmap = LinearSegmentedColormap.from_list('Blues', color_range, N=num_colors)

    return cmap
    
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        
    or just in the main code nans, x= np.isnan(y), lambda z: z.nonzero()[0]
    """

    return np.isnan(y), lambda z: z.nonzero()[0]  
    
def find_last_abrupt_change(data, min_length, tolerance, yearbeg):
    """
    Finds segments of at least `min_length` consecutive elements with the same value,
    allowing for a tolerance of `tolerance` different values within that period.

    :param data: List of values
    :param min_length: Minimum length of consecutive elements
    :param tolerance: Allowed number of deviations
    :return: List of tuples where each tuple is (start_index, end_index) of a segment
    """
    n = len(data)
    segments = []

    start = 0
    while start < n:
        end = start
        deviations = 0
        while end < n:
            if data[end] != data[start]:
                deviations += 1
                if deviations/(end-start) > tolerance/min_length:
                    break
            end += 1
        if end - start >= min_length:
            segments.append((start, end - 1))
        start = end
     
    if not segments:  # Check if stable_slices is empty
        segments.append((-1-yearbeg, - 1-yearbeg))
    start_year = segments[-1][0] + yearbeg
    end_year = segments[-1][1] + yearbeg
    
    #print("start_year",start_year)
    #print("end_year",start_year)
    return start_year, end_year
    
 
# Sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D
    
    
def find_most_common_number(array):

    """
    Trouve le nombre qui apparaît le plus fréquemment dans un tableau
    et calcule le pourcentage de ses occurrences.
    
    Paramètres :
        array (list ou np.ndarray) : Tableau d'entrée contenant les nombres.
    
    Retourne :
        tuple : Le nombre le plus fréquent, son nombre d'occurrences et son pourcentage.
    """
    unique, counts = np.unique(array, return_counts=True)
    max_count_index = np.argmax(counts)
    most_common_number = unique[max_count_index]
    occurrences = counts[max_count_index]
    total_count = len(array)
    percentage = (occurrences / total_count) * 100
    return most_common_number, occurrences, percentage


##############
### CHOICE ###
##############

#0=no, 1=yes 
# input data 
yearbeg=1979
yearend=1998 #1989 #2022
time_step='5D' #'5D' #'M=monthly', 'D=daily', '5D'=5 days mean
n_years=45 #45 ou 1
clim=0 # 1=yes and put n_year=1; 0=no
lat_min=55

# common choice
choix_distance='maha_cor' # #'euclidean', 'maha_cor', 
n_clusters=3 #not taken into account for silhouette =1

#calcul cluster
calcul_cluster=0 
interpol_badvalue=1
remove_0=1 
block_years=15 #no=0= 1 year; 4=block of 4 years, 11=block of 11 years
silhouette=0
low_limit_sc=2
up_limit_sc=3
q_low=0.25
q_high=0.75
coef_fuzzy=2

#diagnostic
plot_diagnostic=1
plot_map=0 
plot_ts=1
plot_regime=1
min_length=10
tolerance=1 

end='_v11.1'
# CHECK _{end}.npy
fontsize=14
dpi=600
format_fig='eps'

print("clim", clim, "n_years", n_years, "time_step:", time_step, "silhouette", silhouette, "n_clusters=", n_clusters)

pathdata_raw="/home/asimon/Documents/WORK/STUDY/POSTDOC_IMT/Arctic_Clustering_Obs_202309/DATA/INPUT/NSIDC"
pathdata_raw_CMIP="/home/asimon/Documents/WORK/STUDY/POSTDOC_IMT/Arctic_Clustering_Obs_202309/DATA/INPUT/CMIP6"
pathdata_mask="/home/asimon/Documents/WORK/STUDY/POSTDOC_IMT/Arctic_Clustering_Obs_202309/DATA/INPUT/Mask_NSIDC-0780"
pathoutput_fig="/home/asimon/Documents/WORK/STUDY/POSTDOC_IMT/Arctic_Clustering_Obs_202309/PLOT"
pathoutput_file="/home/asimon/Documents/WORK/STUDY/POSTDOC_IMT/Arctic_Clustering_Obs_202309/DATA/OUTPUT"


file_name= f'{pathdata_raw}/seaice_conc_monthly_nh_forlonlat.nc'
print(file_name)
fh = Dataset(file_name, mode='r')  
xgrid = fh.variables['xgrid'][:]
ygrid = fh.variables['ygrid'][:]
lat = fh.variables['latitude'][:]
lon = fh.variables['longitude'][:]
fh.close()

#cell_area=25*25 #km2
file_name= f'{pathdata_raw}/seaice_conc_gridarea.nc'
print(file_name)
fh = Dataset(file_name, mode='r')  
cell_area = fh.variables['cell_area'][:]
fh.close()
total_area=np.sum(cell_area)

######################
### CALCUL CLUSTER ###
######################
if calcul_cluster == 1:
   ### INPUT DATA  

   if time_step == 'M':
      nb_time_step=12
      if clim == 0:
         file_name= f'{pathdata_raw}/seaice_conc_monthly_nh_197901202312.nc'
         #!!file_name= f'{pathdata_raw_CMIP}/siconc_SImon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_197901-201412_above40N.nc'
      else:
         file_name= f'{pathdata_raw}/seaice_conc_monthly_clim_nh_197901202312.nc'
      print("filename",file_name)
      fh = Dataset(file_name, mode='r')  
      var= fh.variables['cdr_seaice_conc_monthly'][:,:,:]  # (time,y,x)
      #!!var= fh.variables['siconc'][:,:,:]  # (time,y,x)
      print(file_name)
      print("np.shape(var)",np.shape(var))
      fh.close()
   elif time_step == '5D':
      nb_time_step=73
      if clim == 0:
         file_name= f'{pathdata_raw}/seaice_conc_5days_nh_1979010120231231_del29feb.nc'
      else:
         #file_name= f'{pathdata_raw}/seaice_conc_5days_clim_nh_1979010120231231_del29feb.nc'
         file_name= f'{pathdata_raw}/seaice_conc_5days_clim_nh_1979010119981231_del29feb.nc'   
      fh = Dataset(file_name, mode='r')  
      var= fh.variables['cdr_seaice_conc'][:,:,:]  # (time,y,x)
      print(file_name)
      fh.close()
   elif time_step == 'D':
      nb_time_step=365
      if clim == 0:
         print("open")
         file_name= f'{pathdata_raw}/seaice_conc_daily_nh_1979010120231231_del29feb.nc'
         print("close")
      else:
         file_name= f'{pathdata_raw}/seaice_conc_daily_clim_nh_1979010120231231_del29feb.nc'   
      fh = Dataset(file_name, mode='r')  
      var= fh.variables['cdr_seaice_conc'][:,:,:]  # (time,y,x)
      print(file_name)
      fh.close()  

   print("np.shape(cell_area)",np.shape(cell_area))
   print("shape var",np.shape(var))
   print("shape cell_area",np.shape(cell_area))
   print("shape xgrid",np.shape(xgrid))
   print("shape ygrid",np.shape(ygrid))
   print("shape lon",np.shape(lon))
   print("shape lat",np.shape(lat))
   
   ###########################
   ### CLIMATO FOR SIC=15% ###
   ###########################

   print("CALCULATE CLIMATOLOGY")
   var_clim_x_y=np.zeros((var.shape[1], var.shape[2],nb_time_step))
   for i_timestep in range(nb_time_step):
      var_clim_x_y[:,:,i_timestep]=np.nanmean(var[i_timestep::nb_time_step,:,:], axis=0)              

if calcul_cluster == 1: 

   ### BAD_VALUE 
  
   if interpol_badvalue == 1 :
      print("INTERPOLATION MISSING VALUE")

      var_ = var.filled()
      var_[var.mask] = np.nan
      var=var_
    
      # handle weird case where an ocean pixel become a nan value at another time step
      for x in range(var.shape[2]):
         for y in range(var.shape[1]):
            if np.isnan(var[0,y,x]) == False:
               nans, g= nan_helper(var[:,y,x])
               var[nans,y,x]= np.interp(g(nans), g(~nans), var[~nans,y,x])
               
      var=np.where(lat <lat_min ,np.nan,var)
      var=np.where(var > 200 ,np.nan,var)
      
      
   var_with_0=np.zeros((var.shape[0],var.shape[1],var.shape[2]))
   var_with_0=var.copy()   
   np.save(f'{pathoutput_file}/var_with_0_{yearbeg}{yearend}{time_step}.npy', var_with_0)
   
   ### REMOVE 0 
   if remove_0 == 1:
      print("REMOVE ALL POINTS WITH SIC=0")  
       
      var_0_map=np.zeros((var.shape[1], var.shape[2]))
      var_0=np.zeros((n_years,var.shape[1], var.shape[2]))
      var_land=np.zeros((var.shape[1], var.shape[2]))    
      var_land=np.where(np.isnan(var[0,:,:]) == True, 1, 0)
      np.save(f'{pathoutput_file}/var_land_{yearbeg}{yearend}{time_step}_{choix_distance}.npy', var_land)

      for i_year in range(0,n_years):
         for x in range(var.shape[2]):
            for y in range(var.shape[1]):
               var_0[i_year,y,x]=np.where(np.sum(var[i_year*nb_time_step:i_year*nb_time_step+nb_time_step,y,x],axis=0) == 0., 1., 0.)
               var[i_year*nb_time_step:i_year*nb_time_step+nb_time_step,y,x]=np.where(np.sum(var[i_year*nb_time_step:i_year*nb_time_step+nb_time_step,y,x],axis=0) == 0., np.nan,  var[i_year*nb_time_step:i_year*nb_time_step+nb_time_step,y,x])
               
   np.save(f'{pathoutput_file}/var_{yearbeg}{yearend}{time_step}_{choix_distance}.npy', var)
   
   # Figure article 3b
   for k in range(2, 7):  
         quantiles=np.zeros((nb_time_step,k))
         var_year_month_x_y=var.reshape(-1,nb_time_step,var.shape[1],var.shape[2]) #Reshape the 1D array into a 2D array with 12 rows (months) and the appropriate number of columns
         print("var_year_month_x_y",np.shape(var_year_month_x_y))
         quantile_separation=np.linspace(0,1,k+2)
         #quantile_separation=quantile_separation[1:-1]
         quantile_separation=quantile_separation[1:]
         print("quantile_separation",quantile_separation)
         for mth in range(nb_time_step):
            for i_cluster in range(k):
               quantiles[mth,i_cluster]=np.nanquantile(var_year_month_x_y[:,mth,:,:],quantile_separation[i_cluster])
         print("quantiles",quantiles)
      
         plt.figure(figsize=(8,3))
         plt.grid(linestyle = '--', linewidth = 0.5)
         for i_cluster in range(k):
            plt.plot(np.arange(0,nb_time_step),quantiles[:,i_cluster], 'o-', label=f"Quantile {round(quantile_separation[i_cluster],2)}")
         plt.title(f'Initial cluster centers based on equal quantile separation',fontsize=16)
         if nb_time_step == 12:
            name_months=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May ', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.' ]
            plt.xticks(range(len(name_months)), name_months, fontsize=fontsize)
         else:
            plt.xticks(fontsize=fontsize)   
         plt.yticks(fontsize=fontsize)
         plt.ylabel("SIC (-)", fontsize=fontsize)
         plt.grid('both')
         plt.legend()
         plt.tight_layout()
         plt.savefig(f'{pathoutput_fig}/initial_centroid_quantiles_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{k}_{end}.{format_fig}', format=f'{format_fig}')
         plt.close()        
           
   ### PRE-CLUSTERING 

   print("PRE-CLUSTERING")  
   my_array_withnan = np.zeros((ygrid.shape[0]*xgrid.shape[0],nb_time_step))
   var_nb_time_step_x_y=np.zeros((nb_time_step,var.shape[1], var.shape[2]))
   var_nb_time_step_xy=np.zeros((ygrid.shape[0]*xgrid.shape[0],nb_time_step))   
   var_nb_time_step_x_y[:,:,:]=var[0:nb_time_step,:,:]
   #print("var_nb_time_step_x_y",var_nb_time_step_x_y[:,200,200])
   for t in range(nb_time_step):
      var_nb_time_step_xy[:,t]=var_nb_time_step_x_y[t,:,:].reshape(ygrid.shape[0]*xgrid.shape[0])
   my_array_withnan[:,:]=var_nb_time_step_xy[:,:]

   for i_year in range(1,n_years):
      var_nb_time_step_x_y=np.zeros((nb_time_step,var.shape[1], var.shape[2]))
      var_nb_time_step_x_y[:,:,:]=var[nb_time_step*i_year:nb_time_step*i_year+nb_time_step,:,:]
      
      #print("var_nb_time_step_x_y",var_nb_time_step_x_y[:,200,200])
      for t in range(nb_time_step):
         var_nb_time_step_xy[:,t]=var_nb_time_step_x_y[t,:,:].reshape(ygrid.shape[0]*xgrid.shape[0])
      my_array_withnan=np.concatenate((my_array_withnan,var_nb_time_step_xy),axis=0)

   mask_array_notnan = np.zeros((ygrid.shape[0]*xgrid.shape[0]), dtype=int)
   mask_array_notnan=np.where(np.isnan(var_nb_time_step_xy[:,0]) == True, 0, 1)
   my_array = my_array_withnan[~np.isnan(my_array_withnan).any(axis=1)]
   
   np.save(f'{pathoutput_file}/array_preclustering_{yearbeg}{yearend}{time_step}_{choix_distance}.npy', my_array)
   
### CLUSTERING 

   if silhouette == 1:
      print("silhouette == 1")
      # Silhouette coefficient to choose the number of cluster 
      
      silhouette_samples_value=[]
      #Notice you start at 2 clusters for silhouette coefficient
      for k in range(low_limit_sc, up_limit_sc):
         if choix_distance == 'euclidean': 
            kmeans_ = KMeansEuclidean(k)
         elif choix_distance == 'maha_cor':
            kmeans_ = KMeansMaha_cor(k)        
         
         quantiles=np.zeros((nb_time_step,k))
         var_year_month_x_y=var.reshape(-1,nb_time_step,var.shape[1],var.shape[2]) #Reshape the 1D array into a 2D array with 12 rows (months) and the appropriate number of columns
         quantile_separation=np.linspace(0,1,k+2)
         quantile_separation=quantile_separation[1:]
         print("quantile_separation",quantile_separation)
         for mth in range(nb_time_step):
            for i_cluster in range(k):
               quantiles[mth,i_cluster]=np.nanquantile(var_year_month_x_y[:,mth,:,:],quantile_separation[i_cluster])
         print("quantiles",quantiles)
      
         # Figure article (3b)
         plt.figure(figsize=(8,3))
         plt.grid(linestyle = '--', linewidth = 0.5)
         for i_cluster in range(k):
            plt.plot(np.arange(0,nb_time_step),quantiles[:,i_cluster], 'o-', label=f"Quantile {round(quantile_separation[i_cluster],2)}")
         plt.title(f'Initialization: first centroids for {k} clusters')
         if nb_time_step == 12:
            name_months=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May ', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.' ]
            plt.xticks(range(len(name_months)), name_months, fontsize=fontsize)
         else:
            plt.xticks(fontsize=fontsize)   
         plt.yticks(fontsize=fontsize)
         plt.ylabel("SIC (-)", fontsize=fontsize)
         plt.grid('both')
         plt.legend()
         plt.tight_layout()
         plt.savefig(f'{pathoutput_fig}/initial_centroid_quantiles_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{k}_{end}.{format_fig}', format=f'{format_fig}') 
         plt.close()
         
         kmeans_labels, kmeans_centroids, convergence_centroids = kmeans_.fit(my_array, initialization_centroid="quantile", quantiles=quantiles.T)
         
         print("calcul silhouette")   
         if choix_distance == 'euclidean':
            silhouette_samples_value.append(silhouette_samples(my_array, kmeans_labels, metric='euclidean'))
         elif choix_distance == 'maha_cor':
             #print("before VI")
             #VI = np.linalg.inv(np.corrcoef(my_array.T))
             #print("after VI")
             #silhouette_samples_value.append(silhouette_samples(my_array, kmeans_labels, metric='mahalanobis', VI=VI))
             silhouette_samples_value.append(silhouette_samples(my_array, kmeans_labels, metric='euclidean'))
             print("silhouette_samples_value")
             
      #print("silhouette_samples_value",silhouette_samples_value)  
      mean_silhouette_values = [np.mean(values) for values in silhouette_samples_value]
      print("mean_silhouette_values", mean_silhouette_values)
      median_silhouette_values = [np.median(values) for values in silhouette_samples_value]
      print("median_silhouette_values", median_silhouette_values)
      max_index = np.argmax(mean_silhouette_values)
      n_clusters = max_index + low_limit_sc
      print("n_clusters silhouette", n_clusters)
      np.save(f'{pathoutput_file}/silhouette_coef_mean_{low_limit_sc}{up_limit_sc}_{yearbeg}{yearend}{time_step}_{choix_distance}.npy', mean_silhouette_values)
      np.save(f'{pathoutput_file}/silhouette_coef_median_{low_limit_sc}{up_limit_sc}_{yearbeg}{yearend}{time_step}_{choix_distance}.npy', median_silhouette_values)
      
      # Figure article 3a   
      print("np.shape(silhouette_samples_values)",np.shape(silhouette_samples_value)) 
      plt.figure(figsize=(9, 4))
      bp = plt.boxplot(silhouette_samples_value, positions=range(low_limit_sc, up_limit_sc), whis=(1, 99), showfliers=False, meanline=True, showmeans=True)
      #plt.title("Silhouette coefficient", fontsize=fontsize)
      plt.xlabel(f'Number of clusters', fontsize=fontsize)
      plt.ylabel(f'Silhouette coefficient', fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/silhouette_boxplot_exemple_kmeans_{choix_distance}{end}.{format_fig}', format=f'{format_fig}', dpi=dpi) 
      plt.close()
      
   print("silhouette", silhouette)
   print("n_clusters", n_clusters)
   np.save(f'{pathoutput_file}/n_clusters_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', n_clusters)

   ### kmeans cluster optimum 
   if choix_distance == 'euclidean': 
      kmeans_ = KMeansEuclidean(n_clusters)
   elif choix_distance == 'maha_cor':
      kmeans_ = KMeansMaha_cor(n_clusters)   

   quantiles=np.zeros((nb_time_step,n_clusters))
   var_year_month_x_y=var.reshape(-1,nb_time_step,var.shape[1],var.shape[2]) #Reshape the 1D array into a 2D array with 12 rows (months) and the appropriate number of columns
   quantile_separation=np.linspace(0,1,n_clusters+2)
   quantile_separation=quantile_separation[1:-1]
   for t in range(nb_time_step):
      for i_cluster in range(n_clusters):
         quantiles[t,i_cluster]=np.nanquantile(var_year_month_x_y[:,t,:,:],quantile_separation[i_cluster])
   
   name_months=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May ', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.' ]
   plt.figure(figsize=(8,3))
   plt.grid(linestyle = '--', linewidth = 0.5)
   for i_cluster in range(n_clusters):
      plt.plot(np.arange(0,nb_time_step),quantiles[:,i_cluster], 'o-', label=f"Quantile {round(quantile_separation[i_cluster],2)}")
   plt.title(f'Initialization: first centroids for {n_clusters} clusters', fontsize=fontsize)
   if nb_time_step == 12:
      plt.xticks(range(len(name_months)), name_months, fontsize=fontsize)
   else:
      interval=int(nb_time_step/12)
      positions = [interval-interval/2, 2*interval-interval/2, 3*interval-interval/2, 4*interval-interval/2, 5*interval-interval/2, 6*interval-interval/2, 7*interval -interval/2, 8*interval-interval/2, 9*interval-interval/2, 10*interval-interval/2, 11*interval-interval/2, 12*interval-interval/2]
      plt.xticks(ticks=positions, labels=name_months, fontsize=fontsize)
   plt.yticks(fontsize=fontsize)
   plt.ylabel("SIC (-)", fontsize=fontsize)
   plt.grid('both')
   plt.legend()
   plt.tight_layout()
   plt.savefig(f'{pathoutput_fig}/initial_centroid_quantiles_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}_{end}.{format_fig}', format=f'{format_fig}')
   plt.close()
         
   kmeans_labels, kmeans_centroids, convergence_centroids = kmeans_.fit(my_array, initialization_centroid="quantile", quantiles=quantiles.T)
   
   np.save(f'{pathoutput_file}/cluster_convergence_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', convergence_centroids)
   np.save(f'{pathoutput_file}/kmeans_centroids_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', kmeans_centroids)
     
   centroid_sum_cluster = np.zeros((n_clusters))
   for i_cluster in range(0,n_clusters):
      centroid_sum_cluster[i_cluster]=np.sum(kmeans_centroids[i_cluster,:])
   order_centroid=np.argsort(centroid_sum_cluster)

   kmeans_labels_order = np.empty((np.shape(kmeans_labels)), dtype=int)
   kmeans_labels_order = np.nan
   for i_cluster in range(0,n_clusters):
      kmeans_labels_order = np.where(kmeans_labels == i_cluster , np.where(order_centroid == i_cluster)[0], kmeans_labels_order)

   print("number points per cluster", np.unique(kmeans_labels,return_counts=True))

   ### RE-MAP AND MEAN PROBABILITY

   # MAP WITH YEAR
   year_1d_nan=np.ones((n_years, ygrid.shape[0]*xgrid.shape[0]))*np.nan
   
   #reintroduce land and ocean only
   count_kmeans=0
   for i_year in range(0,n_years):
      count=0
      for y in range(var.shape[1]):
         for x in range(var.shape[2]):
            if var_0[i_year,y,x] == 1:
               year_1d_nan[i_year,count]=0
               count += 1
            elif var_land[y,x] == 1:
               year_1d_nan[i_year,count]=np.nan
               count += 1
            else:
               year_1d_nan[i_year,count]=kmeans_labels_order[count_kmeans]+1
               count += 1
               count_kmeans += 1
                     
   year_map_nan=np.zeros((n_years,var.shape[1], var.shape[2]))
   year_map_nan=year_1d_nan.reshape(n_years, var.shape[1], var.shape[2])

   np.save(f'{pathoutput_file}/cluster_map_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', year_map_nan)

   ### SEASONAL CYCLE 
   mean_SIC_clusters=np.zeros((n_clusters+1,nb_time_step))

   mean_SIC_clusters[0,:]=0

   print("mean_SIC_clusters",*mean_SIC_clusters)

   for i_cluster in range(0,n_clusters):
      kmeans_true_cluster=np.where(kmeans_labels_order==i_cluster,True, False)
      for i_timestep in range(nb_time_step):
         mean_SIC_clusters[i_cluster+1,i_timestep]=np.nanmean(my_array[:,i_timestep],where=kmeans_true_cluster)

         
   np.save(f'{pathoutput_file}/cluster_centroid_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', mean_SIC_clusters)
   np.save(f'{pathoutput_file}/kmeans_labels_order_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy',kmeans_labels_order)
   
   ### MAP OF PROBABILITY 
   print("CALCUL OF DISTANCE FOR PROBABILITY",np.shape(var_with_0))
   distance_=np.zeros((n_years, n_clusters+1, var.shape[1], var.shape[2]))
   map_proba=np.zeros((n_years, n_clusters+1, var.shape[1], var.shape[2]))

   inv_cov = np.linalg.inv(np.cov(my_array.T))
   inv_corr = np.linalg.inv(np.corrcoef(my_array.T))
   var_euclidean = np.var(my_array.T, axis=1) 
   year_indices = [range(nb_time_step*i_year, nb_time_step*i_year+nb_time_step) for i_year in range(n_years)]

   total_area_withoutland=0.
   for x in range(var.shape[2]):
      for y in range(var.shape[1]):
         if var_land[y,x] == 0:
            total_area_withoutland += cell_area[y,x]
            for i_year in range(0,n_years):
               yearly_data = var_with_0[year_indices[i_year], y, x]
               for i_cluster in range(0,n_clusters+1):
                  mean_cluster = mean_SIC_clusters[i_cluster, :]
                  distance_[i_year,i_cluster,y,x]=distance.euclidean(yearly_data, mean_cluster)
               
     
               for i_cluster in range(0,n_clusters+1):
                  map_proba[i_year,i_cluster,y,x]=1./(np.nansum((distance_[i_year,i_cluster,y,x]/distance_[i_year,:,y,x])**(2/(coef_fuzzy-1))))
                  if np.isinf(map_proba[i_year,i_cluster,y,x]):
                     map_proba[i_year,i_cluster,y,x]=1.
         else:
            map_proba[:,:,y,x]=np.nan
                      
   np.save(f'{pathoutput_file}/cluster_map_proba_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy', map_proba)
   
   del distance_
   del var
   del var_with_0
   del total_area_withoutland
   del year_indices
   del inv_cov
   del inv_corr
   del var_euclidean
   del kmeans_true_cluster
   del year_1d_nan
   del my_array
   del kmeans_labels
   del kmeans_labels_order
   del kmeans_centroids
   del order_centroid
   
   del convergence_centroids
   del mean_SIC_clusters
   del year_map_nan
   del map_proba


##################
### DIAGNOSTIC ###
##################

if plot_diagnostic == 1:

   ### LOAD
   convergence_centroids=np.load(f'{pathoutput_file}/cluster_convergence_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   mean_SIC_clusters=np.load(f'{pathoutput_file}/cluster_centroid_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   year_map_nan=np.load(f'{pathoutput_file}/cluster_map_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   map_proba = np.load(f'{pathoutput_file}/cluster_map_proba_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   var = np.load(f'{pathoutput_file}/var_{yearbeg}{yearend}{time_step}_{choix_distance}.npy')
   var_with_0 = np.load(f'{pathoutput_file}/var_with_0_{yearbeg}{yearend}{time_step}.npy')
   var_land = np.load(f'{pathoutput_file}/var_land_{yearbeg}{yearend}{time_step}_{choix_distance}.npy')
   array_preclustering=np.load(f'{pathoutput_file}/array_preclustering_{yearbeg}{yearend}{time_step}_{choix_distance}.npy')
   kmeans_labels_order=np.load(f'{pathoutput_file}/kmeans_labels_order_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   kmeans_centroids = np.load(f'{pathoutput_file}/kmeans_centroids_{yearbeg}{yearend}{time_step}_{choix_distance}_nclu{n_clusters}.npy')
   
   print("np.shape(convergence_centroids)",np.shape(convergence_centroids))
   print("np.shape(mean_SIC_clusters)",np.shape(mean_SIC_clusters))
   print("np.shape(year_map_nan)",np.shape(year_map_nan))
   print("np.shape(map_proba)",np.shape(map_proba))
   print("np.shape(var)",np.shape(var))
   print("np.shape(var_with_0)",np.shape(var_with_0))
   print("np.shape(var_land)",np.shape(var_land))
   print("np.shape(array_preclustering)",np.shape(array_preclustering))
   print("np.shape(kmeans_centroids)",np.shape(kmeans_centroids))
   
   nb_time_step=mean_SIC_clusters.shape[1]
                    
   ### COLOR AND NAME
         
   new_cmap = plt.cm.get_cmap('rocket')
   num_segments = n_clusters+1
   segmented_colors = [new_cmap(i/ num_segments) for i in range(num_segments)]
  
   print("segmented_colors",segmented_colors)
   print("segmented_colors",type(segmented_colors))
   

   blue_color = (0.14, 0.47, 1.0, 1.0) 
   gray_color = (0.35, 0.35, 0.35, 1.0)
   purple_color= (0.76, 0.56, 0.67, 1.0)
   orange_color = (0.97, 0.69, 0.63, 1.0)
   segmented_colors[0]=blue_color
   segmented_colors[1]=orange_color
   segmented_colors[3]=segmented_colors[2]
   segmented_colors[2]=gray_color
   
   # Define a color map for the dominant_cluster values
   color_cluster = [
   blue_color,
   gray_color,
   purple_color,
   orange_color
]

   segmented_cmap = ListedColormap(segmented_colors)
   print("segmented_colors",segmented_colors)

   segmented_hex_colors = [to_hex(color)[1:] for color in segmented_colors]
   print("segmented_hex_colors",segmented_hex_colors)

   if n_clusters == 3:
      i_cluster_name=["Open-ocean cluster", "Partial winter-freezing cluster", "Full winter-freezing cluster ", "Permanent sea-ice cluster "]
      order_cluster_name = [3, 2, 1, 0]  # Custom order, you can set it as per your requirements
   else:
      i_cluster_name=np.arange(n_clusters+1)
      order_cluster_name = np.arange(n_clusters+1)
   print("i_cluster_name",i_cluster_name)
   
   name_months=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May ', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.' ]
   
   marker_cluster = (['o-', 's-', 'd-', 'p-']) 
   marker_cluster_2 = (['o', 's', 'd', 'p'])    
   
   ### MATRIX OF CORRELATION - Figure article 2b
   fig = plt.figure(figsize=(6,6))
   fig.subplots_adjust(right=20)
   plt.matshow(np.corrcoef(array_preclustering.T), cmap='RdBu_r',vmin=-1,vmax=1)
   if nb_time_step == 12:
      plt.xticks(range(len(name_months)), name_months, fontsize=9)
      plt.yticks(range(len(name_months)), name_months, fontsize=9)
   else:
      interval=int(nb_time_step/12)
      positions = [interval-interval/2, 2*interval-interval/2, 3*interval-interval/2, 4*interval-interval/2, 5*interval-interval/2, 6*interval-interval/2, 7*interval -interval/2, 8*interval-interval/2, 9*interval-interval/2, 10*interval-interval/2, 11*interval-interval/2, 12*interval-interval/2]
      plt.xticks(ticks=positions, labels=name_months, fontsize=9) 
      plt.yticks(ticks=positions, labels=name_months, fontsize=9)
   plt.grid(False)
   plt.title(f' Correlation matrix') 
   plt.tight_layout()
   plt.colorbar(fraction=0.046, pad=0.04)
   plt.savefig(f'{pathoutput_fig}/matrix_correlation_{choix_distance}.{format_fig}', bbox_inches="tight", format=f'{format_fig}',dpi=dpi)
   plt.close('all')  
      
   
   ### CONVERGENCE
   fig = plt.figure(figsize=(6,6))
   for i_cluster in range(0,n_clusters):
      plt.plot(convergence_centroids[i_cluster,:], 'o-', label=f'Cluster {i_cluster_name[i_cluster+1]}', c=segmented_cmap(i_cluster+1), alpha=1)
   plt.ylim([-1, 1])
   plt.tight_layout()
   plt.legend()
   plt.savefig(f'{pathoutput_fig}/convergence_centroids_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)
   plt.close('all')
   
   ### HISTOGRAM -  figure review report  
   fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
   axes = axes.flatten()  # Make it easier to index

   for mth, t in enumerate(range(3, 15)):  # 12 months
       ax = axes[mth]
       ax.hist(array_preclustering[:, t + 6], bins=50, edgecolor='black')
       ax.set_title(f'15th of month {mth + 1}')
       ax.set_xlabel('Value')
       ax.set_ylabel('Frequency')
       ax.grid(True)
   # Hide any unused subplots (just in case)
   for i in range(len(range(3, 15)), len(axes)):
       fig.delaxes(axes[i])
   fig.suptitle('Histograms of sea-ice concentration data used for clustering', fontsize=16)
   fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
   fig.savefig(f'{pathoutput_fig}/histogram_15th_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi) 
   
   ### SEASONAL CYCLE  
   # Figure article 4b
   plt.figure(figsize=(8,4))
   #plt.plot(np.arange(1,nb_time_step+1), mean_SIC_clusters[0,:], marker='o',  linestyle='solid', linewidth=2, color='blue', label=f'Cluster {i_cluster_name[0]}', alpha=1)
   for i_cluster in range(0,n_clusters+1):
      plt.plot(np.arange(0,nb_time_step),mean_SIC_clusters[i_cluster,:], '-', c=segmented_cmap(i_cluster), alpha=1, linewidth=3, label=f'{i_cluster_name[i_cluster]}')
   #for i_year in range(0,n_years):
   #   plt.plot(np.arange(0,nb_time_step),var[nb_time_step*i_year:nb_time_step*i_year+nb_time_step,200,200], '.--', alpha=0, color="white", linewidth=1) #label=f"Year {i_year+yearbeg}"
   plt.grid(linestyle = '--', linewidth = 0.5)
   #plt.title(f'Mean seasonal cycle of SIC with all data from {yearbeg}-{yearend} - {choix_distance}')
   # Apply the sorted order to the legend
   plt.grid('both')
   plt.yticks(fontsize=fontsize)
   interval=int(nb_time_step/12)
   positions = [interval-interval/2, 2*interval-interval/2, 3*interval-interval/2, 4*interval-interval/2, 5*interval-interval/2, 6*interval-interval/2, 7*interval -interval/2, 8*interval-interval/2, 9*interval-interval/2, 10*interval-interval/2, 11*interval-interval/2, 12*interval-interval/2]
   plt.xticks(ticks=positions, labels=name_months, fontsize=fontsize) 
   plt.ylabel('Sea-ice concentration', fontsize=fontsize)
   handles, labels = plt.gca().get_legend_handles_labels()
   plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], loc=(0.0, 0.07), fontsize=12)
   plt.margins(x=0.01)
   plt.tight_layout()
   if clim == 0:
      plt.savefig(f'{pathoutput_fig}/seasonal_cycle_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', bbox_inches='tight', format=f'{format_fig}')
   else:
      plt.savefig(f'{pathoutput_fig}/seasonal_cycle_clim_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', bbox_inches='tight', format=f'{format_fig}')
   plt.close('')
    
   # with spread - Figure article S2
   # Initialize arrays
   cluster_proba_max = np.zeros((n_years, var.shape[1], var.shape[2]), dtype=int)
   mask_cluster_max = np.zeros((n_years, n_clusters + 1, var.shape[1], var.shape[2]), dtype=bool)

   for i_cluster in range(n_clusters + 1):
       for x in range(var.shape[2]):
           for y in range(var.shape[1]):
               if var_land[y, x] == 0:
                   for i_year in range(n_years):
                       cluster_proba_max[i_year, y, x] = np.argmax(map_proba[i_year, :, y, x])
                       mask_cluster_max[i_year, i_cluster, y, x] = (cluster_proba_max[i_year, y, x] == i_cluster)

   var_with_0_reshaped = var_with_0.reshape(n_years, nb_time_step, var.shape[1], var.shape[2])
   quantile_spread_high_clusters=np.zeros((n_clusters+1,nb_time_step))
   quantile_spread_low_clusters=np.zeros((n_clusters+1,nb_time_step))
   for i_cluster in range(0,n_clusters+1):
      indices = np.where(mask_cluster_max[:, i_cluster, :, :])
      print("indices",indices)
      for i_timestep in range(nb_time_step):
         selected_data=var_with_0_reshaped[indices[0], i_timestep, indices[1], indices[2]]
         quantile_spread_high_clusters[i_cluster,i_timestep]=np.nanquantile(selected_data,0.90)
         quantile_spread_low_clusters[i_cluster,i_timestep]=np.nanquantile(selected_data,0.10)
   
   
   median_SIC_clusters=np.zeros((n_clusters+1,nb_time_step))   
   for i_cluster in range(0,n_clusters):
      kmeans_true_cluster=np.where(kmeans_labels_order==i_cluster,True, False)
      for i_timestep in range(nb_time_step):
         filtered_array = array_preclustering[kmeans_true_cluster, i_timestep]
         median_SIC_clusters[i_cluster+1, i_timestep] = np.nanquantile(filtered_array, 0.5)
         
   
   plt.figure(figsize=(8,4))
   for i_cluster in range(0,n_clusters+1):
      plt.plot(np.arange(0,nb_time_step),median_SIC_clusters[i_cluster,:], '-', c=segmented_cmap(i_cluster), alpha=1, linewidth=3, label=f'{i_cluster_name[i_cluster]}')
      plt.plot(np.arange(0,nb_time_step),quantile_spread_high_clusters[i_cluster,:],  linestyle='dashed', c=segmented_cmap(i_cluster), alpha=1, linewidth=1)
      plt.plot(np.arange(0,nb_time_step),quantile_spread_low_clusters[i_cluster,:],  linestyle='dashed', c=segmented_cmap(i_cluster), alpha=1, linewidth=1)
   plt.grid(linestyle = '--', linewidth = 0.5)
   plt.grid('both')
   plt.yticks(fontsize=fontsize)
   interval=int(nb_time_step/12)
   positions = [interval-interval/2, 2*interval-interval/2, 3*interval-interval/2, 4*interval-interval/2, 5*interval-interval/2, 6*interval-interval/2, 7*interval -interval/2, 8*interval-interval/2, 9*interval-interval/2, 10*interval-interval/2, 11*interval-interval/2, 12*interval-interval/2]
   plt.xticks(ticks=positions, labels=name_months, fontsize=fontsize) 
   plt.ylabel('Sea-ice concentration', fontsize=fontsize)
   handles, labels = plt.gca().get_legend_handles_labels()
   plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], loc=(0.0, 0.07), fontsize=12)
   plt.margins(x=0.01)
   plt.tight_layout()
   if clim == 0:
      plt.savefig(f'{pathoutput_fig}/seasonal_cycle_cluster{n_clusters}_spreadmedian_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', bbox_inches='tight', format=f'{format_fig}')
   else:
      plt.savefig(f'{pathoutput_fig}/seasonal_cycle_clim_cluster{n_clusters}_spreadmedian_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', bbox_inches='tight', format=f'{format_fig}')
   plt.close('')    
   
                    
   sic_threshold_min = 0.1
   sic_threshold_max = 0.9
   march_1st_offset = 0 #59 #for timestep=5)
   
   var_with_0_buf = var_with_0 # to start in January with 5d time step
   #var_with_0_buf = var_with_0[12:var.shape[0]-61,:,:] # to start in march with 5d time step
   #var_with_0_buf = var_with_0[30:var.shape[0]-43,:,:] # to start in May with 5d time step
   #var_with_0_buf = var_with_0[43:var.shape[0]-30,:,:] # to start in August with 5d time step
   n_years_init=n_years
   #n_years = n_years-1 # to uncomment if start is not January
   #var_with_0_repet5 = np.repeat(var_with_0, repeats=5, axis=0) #to remove
   
   from scipy.ndimage import uniform_filter1d

   # Example smoothing along the first axis (time axis) with a 15-step moving average
   var_with_0_repet5_smooth = uniform_filter1d(var_with_0_buf, size=3, axis=0, mode='nearest')
   var_year_nb_time_step_x_y = var_with_0_repet5_smooth.reshape(n_years, nb_time_step, var.shape[1], var.shape[2])


   var_year_nb_time_step_x_y_offset = np.roll(var_year_nb_time_step_x_y, -march_1st_offset, axis=1)
   
   first_retreat_day = np.full((n_years, var.shape[1], var.shape[2]), np.nan) #the first time step after the maximum SIC that is below 0.8
   last_retreat_day = np.full((n_years, var.shape[1], var.shape[2]), np.nan)
   first_advance_day = np.full((n_years, var.shape[1], var.shape[2]), np.nan)   
   
   # Find date of melting and freezing
   for year in range(n_years):
       for x in range(var.shape[2]):
           for y in range(var.shape[1]):
               if var_land[y, x] == 0:
                   time_series = var_year_nb_time_step_x_y_offset[year, :, y, x]

                   # Find the index of the minimum SIC
                   min_sic_index = np.argmin(time_series)
                   max_sic_index = np.argmax(time_series)


                   # Find the first timestep after the minimum SIC that is above the threshold (FAD)
                   if np.any(time_series < 0.01) :
                      above_threshold_candidates = np.where(time_series[min_sic_index:] > sic_threshold_min)[0]

                      if above_threshold_candidates.size > 0:
                          first_advance_day[year, y, x] = min_sic_index + above_threshold_candidates[0]

                      # Find the last timestep before the minimum SIC that is below the threshold (LRD)
                      below_threshold_candidates = np.where(time_series[:min_sic_index] < sic_threshold_min)[0]
                      if below_threshold_candidates.size > 0:
                          last_retreat_day[year, y, x] = below_threshold_candidates[-1]
                      else:
                          last_retreat_day[year, y, x]= min_sic_index
                   
                   if np.any(time_series > 0.99) : #select time series with full ice
                      below_threshold_max_candidates = np.where(time_series[max_sic_index:] < sic_threshold_max)[0]
                      if below_threshold_max_candidates.size > 0:
                          first_retreat_day[year, y, x] = max_sic_index + below_threshold_max_candidates[0]
                       
   data_first_retreat_day = first_retreat_day.flatten()
   bins = np.arange(np.nanmin(data_first_retreat_day), np.nanmax(data_first_retreat_day) + 1)
   # Days per month in a non-leap year
   days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
   # Calculate cumulative day indices divided by 5 (to match 5-day averages)
   month_start_days = np.cumsum(np.insert(days_in_month, 0, 0))[:-1]  # Days at start of each month
   #month_start_days = np.cumsum(np.insert(1, 0, 0))[:-1]  # Days at start of each month
   month_start_steps = (month_start_days / 5).astype(int)
   # Labels for each month
   month_labels = ['Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
      
   first_retreat_day = first_retreat_day + march_1st_offset
   last_retreat_day = last_retreat_day +march_1st_offset
   first_advance_day = first_advance_day +march_1st_offset
   
   # Initialize arrays
   cluster_proba_max = np.zeros((n_years, var.shape[1], var.shape[2]), dtype=int)
   mask_cluster_max = np.zeros((n_years, n_clusters + 1, var.shape[1], var.shape[2]), dtype=bool)

   # Collect statistics for each cluster
   first_advance_stats = []
   last_retreat_stats = []
   first_retreat_stats = []
   
   first_retreat_all_clusters = {}
   last_retreat_all_clusters = {}
   first_advance_all_clusters = {}

   for i_cluster in range(n_clusters + 1):
       for x in range(var.shape[2]):
           for y in range(var.shape[1]):
               if var_land[y, x] == 0:
                   for i_year in range(n_years):
                       cluster_proba_max[i_year, y, x] = np.argmax(map_proba[i_year, :, y, x])
                       mask_cluster_max[i_year, i_cluster, y, x] = (cluster_proba_max[i_year, y, x] == i_cluster)

       indices = np.where(mask_cluster_max[:, i_cluster, :, :])
       
       data_cluster_first_advance = first_advance_day[indices[0], indices[1], indices[2]]
       data_cluster_last_retreat = last_retreat_day[indices[0], indices[1], indices[2]]
       data_cluster_first_retreat = first_retreat_day[indices[0], indices[1], indices[2]]
       first_advance_stats.append((
           np.nanquantile(data_cluster_first_advance, 0.75), #because 5 days mean
           np.nanquantile(data_cluster_first_advance,0.5),
           np.nanquantile(data_cluster_first_advance, 0.25)
       ))

       last_retreat_stats.append((
           np.nanquantile(data_cluster_last_retreat, 0.75),
           np.nanquantile(data_cluster_last_retreat,0.5),
           np.nanquantile(data_cluster_last_retreat, 0.25)
       ))
       
       first_retreat_stats.append((
           np.nanquantile(data_cluster_first_retreat, 0.75),
           np.nanquantile(data_cluster_first_retreat,0.5),
           np.nanquantile(data_cluster_first_retreat, 0.25)
       ))
            
       
       #proba using quantile
       # Flatten and clean the data (remove NaNs, Infs, and non-positive values)
       data_cluster_first_retreat_flat = data_cluster_first_retreat.flatten()
       data_cluster_first_retreat_flat = data_cluster_first_retreat_flat[np.isfinite(data_cluster_first_retreat_flat)]  # remove NaN and inf
       
       data_cluster_last_retreat_flat = data_cluster_last_retreat.flatten()
       data_cluster_last_retreat_flat = data_cluster_last_retreat_flat[np.isfinite(data_cluster_last_retreat_flat)]  # remove NaN and inf
       
       data_cluster_first_advance_flat = data_cluster_first_advance.flatten()
       data_cluster_first_advance_flat = data_cluster_first_advance_flat[np.isfinite(data_cluster_first_advance_flat)] 
       
       first_retreat_all_clusters[i_cluster] = data_cluster_first_retreat_flat.copy()
       last_retreat_all_clusters[i_cluster] = data_cluster_last_retreat_flat.copy()
       first_advance_all_clusters[i_cluster] = data_cluster_first_advance_flat.copy()
       
    
   # Define bins and centers
   bins = np.arange(1, 77)
   day_centers = np.arange(0, 75)
   # Assuming first_retreat_all_clusters and other variables are defined
   # Histograms
   counts0, _ = np.histogram(first_retreat_all_clusters[0], bins=bins)
   counts1, _ = np.histogram(first_retreat_all_clusters[1], bins=bins)
   counts2, _ = np.histogram(first_retreat_all_clusters[2], bins=bins)
   counts3, _ = np.histogram(first_retreat_all_clusters[3], bins=bins)   
   total = counts0 + counts1 + counts2 + counts3
   norm0 = (counts0 / total) * 100
   norm1 = (counts1 / total) * 100
   norm2 = (counts2 / total) * 100
   norm3 = (counts3 / total) * 100
   # Stacked areas: compute bottom and top
   bottom0 = np.zeros_like(norm0)
   top0 = norm0
   bottom1 = top0
   top1 = top0 + norm1
   bottom2 = top1
   top2 = top1 + norm2
   bottom3 = top2
   top3 = top2 + norm3  # Should be close to 100
   # Step 1: Compute total sum and target threshold (80%)
   total_sum = np.sum(total)
   threshold = 0.9 * total_sum
   # Step 2: Initialize variables to track shortest period
   min_length_r = len(total) + 1
   start_idx = -1
   end_idx = -1
   # Step 3: Brute-force sliding window approach
   for i in range(len(total)):
       current_sum = 0
       for j in range(i, len(total)):
           current_sum += total[j]
           if current_sum >= threshold:
               if (j - i + 1) < min_length_r:
                   min_length_r = j - i + 1
                   start_idx = i
                   end_idx = j
               break  # No need to keep going, longer windows are worse
   # Step 4: Result
   if start_idx != -1:
       smallest_period = total[start_idx:end_idx+1]
   else:
       print("No such period found.")

   print("Smallest period that fits 80% of the total sum for date of retreat:", smallest_period)

   # Figue article 5a
   fig, ax1 = plt.subplots(figsize=(10, 6))
   # Fill between for the stacked areas
   ax1.fill_between(day_centers, bottom0, top0, color=segmented_cmap(0), label=i_cluster_name[0])
   ax1.fill_between(day_centers, bottom1, top1, color=segmented_cmap(1), label=i_cluster_name[1])
   ax1.fill_between(day_centers, bottom2, top2, color=segmented_cmap(2), label=i_cluster_name[2])
   ax1.fill_between(day_centers, bottom3, top3, color=segmented_cmap(3), label=i_cluster_name[3])
   # Optional: draw black boundaries
   ax1.plot(day_centers, top0, color='black', linewidth=2)
   ax1.plot(day_centers, top1, color='black', linewidth=2)
   ax1.plot(day_centers, top2, color='black', linewidth=2)
   ax1.plot(day_centers, top3, color='black', linewidth=2)
   # Define monthly ticks (assuming each month is 6 days)
   month_positions = np.arange(0, 73-6, 6)  # Tick positions every 6 days
   print("month_positions",month_positions)
   #month_labels = [f'Month {i+1}' for i in range(len(month_positions))]  # Labels for the ticks
   month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
   # Month ticks
   ax1.set_xticks(ticks=month_positions)
   ax1.set_xticklabels(labels=month_labels, rotation=45, fontsize=fontsize)
   ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
   ax1.tick_params(axis='y', labelsize=fontsize)
   # Axis and style for the first y-axis
   ax1.set_xlim(-1, 72)
   ax1.set_ylim(0, 100)
   ax1.set_xlabel('First day of retreat', fontsize=fontsize)
   ax1.set_ylabel('Normalized Percentage (%)', fontsize=fontsize)
   #ax1.set_title('Stacked Distribution of First Retreat Day by Cluster')
   ax1.grid(False)
   # Create a second y-axis for the total variable
   ax2 = ax1.twinx()
   ax2.plot(day_centers[0:72], total[0:72], marker='*', linestyle='-', color='black', label='Total number of first retreat day',zorder=1)
   ax2.scatter(start_idx, total[start_idx], marker='o', facecolors='green', edgecolors='green', zorder=2)
   ax2.scatter(end_idx, total[end_idx], marker='s', facecolors='green', edgecolors='green', zorder=3)   
   ax2.set_ylabel('Total number of first day of retreat', fontsize=fontsize)
   ax2.grid(False)
   ax2.tick_params(axis='y', labelsize=fontsize)
   plt.tight_layout()  
   # Save the plot
   plt.savefig(f'{pathoutput_fig}/stacked_fillbetween_totalnb_all_clusters_first_retreat.{format_fig}', format=format_fig, dpi=dpi)
   

   bins = np.arange(1, 77)
   day_centers = np.arange(0, 75)
   # Assuming first_retreat_all_clusters and other variables are defined
   # Histograms
   counts0, _ = np.histogram(first_advance_all_clusters[0], bins=bins)
   counts1, _ = np.histogram(first_advance_all_clusters[1], bins=bins)
   counts2, _ = np.histogram(first_advance_all_clusters[2], bins=bins)
   counts3, _ = np.histogram(first_advance_all_clusters[3], bins=bins)
   total = counts0 + counts1 + counts2 + counts3
   print("advance total",total)
   norm0 = (counts0 / total) * 100
   norm1 = (counts1 / total) * 100
   norm2 = (counts2 / total) * 100
   norm3 = (counts3 / total) * 100
   # Stacked areas: compute bottom and top
   bottom0 = np.zeros_like(norm0)
   top0 = norm0
   bottom1 = top0
   top1 = top0 + norm1
   bottom2 = top1
   top2 = top1 + norm2
   bottom3 = top2
   top3 = top2 + norm3  # Should be close to 100
   # Step 1: Compute total sum and target threshold (80%)
   total_sum = np.sum(total)
   threshold = 0.9 * total_sum
   # Step 2: Initialize variables to track shortest period
   min_length_a = len(total) + 1
   start_idx = -1
   end_idx = -1
   # Step 3: Brute-force sliding window approach
   for i in range(len(total)):
       current_sum = 0
       for j in range(i, len(total)):
           current_sum += total[j]
           if current_sum >= threshold:
               if (j - i + 1) < min_length_a:
                   min_length_a = j - i + 1
                   start_idx = i
                   end_idx = j
               break  # No need to keep going, longer windows are worse
   # Step 4: Result
   if start_idx != -1:
       smallest_period = total[start_idx:end_idx+1]
   else:
       print("No such period found.")

   print("Smallest period that fits 80% of the total sum for date of advance:", smallest_period)

   # Figure article 5b 
   fig, ax1 = plt.subplots(figsize=(10, 6))
   # Fill between for the stacked areas
   ax1.fill_between(day_centers, bottom0, top0, color=segmented_cmap(0), label=i_cluster_name[0])
   ax1.fill_between(day_centers, bottom1, top1, color=segmented_cmap(1), label=i_cluster_name[1])
   ax1.fill_between(day_centers, bottom2, top2, color=segmented_cmap(2), label=i_cluster_name[2])
   ax1.fill_between(day_centers, bottom3, top3, color=segmented_cmap(3), label=i_cluster_name[3])

   # Optional: draw black boundaries
   ax1.plot(day_centers, top0, color='black', linewidth=2)
   ax1.plot(day_centers, top1, color='black', linewidth=2)
   ax1.plot(day_centers, top2, color='black', linewidth=2)
   ax1.plot(day_centers, top3, color='black', linewidth=2)
   # Month ticks
   month_labels = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
   ax1.set_xticks(ticks=month_positions)
   ax1.set_xticklabels(labels=month_labels, rotation=45, fontsize=fontsize)
   ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
   ax1.tick_params(axis='y', labelsize=fontsize)
   # Axis and style for the first y-axis
   ax1.set_xlim(-1, 72)
   ax1.set_ylim(0, 100)
   ax1.set_xlabel('First day of advance', fontsize=fontsize)
   ax1.set_ylabel('Normalized Percentage (%)',fontsize=fontsize)
   # Create a second y-axis for the total variable
   ax2 = ax1.twinx()
   ax2.plot(day_centers[0:72], total[0:72], marker='*', linestyle='-', color='black', label='Total number of first advance day for all clusters', zorder=1)
   ax2.scatter(start_idx, total[start_idx], marker='o', facecolors='green', edgecolors='green', zorder=2)
   ax2.scatter(end_idx, total[end_idx], marker='s', facecolors='green', edgecolors='green', zorder=3)
   ax2.set_ylabel('Total number of first day of advance', fontsize=fontsize)
   ax2.grid(False)
   ax2.tick_params(axis='y', labelsize=fontsize)
   plt.tight_layout()
   # Save the plot
   plt.savefig(f'{pathoutput_fig}/stacked_fillbetween_totalnb_all_clusters_first_advance.{format_fig}', format=format_fig, dpi=dpi)
  

   ### CLIMATOLOGY
   print("CALCULATE CLIMATOLOGY")
   var_clim_x_y=np.zeros((var.shape[1], var.shape[2],nb_time_step))
   for i_timestep in range(nb_time_step):
      var_clim_x_y[:,:,i_timestep]=np.nanmean(var[i_timestep::nb_time_step,:,:], axis=0)        
   
   ### FREQUENCY
   # map
   freq_cluster_map=np.zeros((n_clusters+1,var.shape[1], var.shape[2]))
   for i_cluster in range(0,n_clusters+1):
      for i_year in range(0,n_years):
         freq_cluster_map[i_cluster,:,:]=np.where(year_map_nan[i_year,:,:] == i_cluster, freq_cluster_map[i_cluster,:,:]+1/(n_clusters+1),  freq_cluster_map[i_cluster,:,:])
         freq_cluster_map[i_cluster,:,:]=np.where(var_land == 1, np.nan,  freq_cluster_map[i_cluster,:,:])      
   for i_cluster in range(0,n_clusters+1):
      start_color = "FFFFFF"
      end_color = segmented_hex_colors[i_cluster]
      color_map_cluster = create_color_range(start_color, end_color)
         
      fig = plt.figure(figsize=(6,6))
      plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      #plt.title(f'SIC Frequency of cluster {i_cluster+1} (total of {n_clusters} clusters) - {choix_distance} {end}')
      plt.contourf(xgrid,ygrid,freq_cluster_map[i_cluster,:,:], cmap=color_map_cluster, alpha=1, levels=np.linspace(0,12,13))
      cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.ax.tick_params(labelsize=18)
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')  
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      plt.tight_layout()
      if clim == 0:
         plt.savefig(f'{pathoutput_fig}/map_freq_cluster{i_cluster+1}_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)  
      else:
         plt.savefig(f'{pathoutput_fig}/map_freq_clim_cluster{i_cluster+1}_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)       
      plt.close('')
   plt.close('')

   # ts
   freq_cluster_ts=np.zeros((n_years, n_clusters+1))
   nb_point_cluster=np.zeros((n_years, n_clusters+1))
   plt.figure(figsize=(8,4))
   for i_year in range(0,n_years):
      for i_cluster in range(0,n_clusters+1):
         nb_point_cluster[i_year,i_cluster]=np.sum(np.where(year_map_nan[i_year,:,:]==i_cluster, 1.,0))
   for i_year in range(0,n_years):
      for i_cluster in range(0,n_clusters+1):
         freq_cluster_ts[i_year,i_cluster] = nb_point_cluster[i_year,i_cluster]/np.sum(nb_point_cluster[i_year,:]) 
   year_range = range(yearbeg, yearbeg + n_years)
   for i_cluster in range(0,n_clusters+1):
      plt.plot(year_range, freq_cluster_ts[:,i_cluster],'o-', label=f'Cluster {i_cluster_name[i_cluster]}', c=segmented_cmap(i_cluster), alpha=1)
   plt.title(f'Transient frequency of each clusters - {choix_distance}{end}')
   plt.ticklabel_format(useOffset=False)
   plt.tight_layout()
   plt.legend()
   plt.xticks(np.arange(yearbeg,yearbeg + n_years,3))
   if clim == 0:
      plt.savefig(f'{pathoutput_fig}/ts_freq_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)
   else:
      plt.savefig(f'{pathoutput_fig}/ts_freq_clim_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)
   plt.close('')
      
   n_years=n_years_init
   if plot_map == 1:
      print("plot_map")
      
      # figure article 4a
      for i_year in range(0,n_years):
         fig = plt.figure(figsize=(6,6))
         ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
         polarCentral_set_latlim((lat_min,90),ax)
         plt.contourf(xgrid,ygrid,year_map_nan[i_year,:,:], cmap=segmented_cmap, levels=np.linspace(-0.01,n_clusters+1,n_clusters+2), alpha=1)
         cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
         cb.ax.tick_params(length=0)
         cb.set_ticks(np.arange(0.5,n_clusters+1+0.5,1))
         #cb.set_ticklabels(np.arange(1,n_clusters+1+1,1))
         cb.set_ticklabels(i_cluster_name)
         plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
         plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')   
         #plt.contour(xgrid,ygrid,var_clim_x_y[:,:,0], linestyles='dotted', colors='black')
         ax.add_feature(cfeature.LAND)
         ax.coastlines(linewidth=0.5,color='k')          
         ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
         fig.set_tight_layout(True)
         if clim == 0:
            plt.savefig(f'{pathoutput_fig}/map_fromclustering_cluster{n_clusters}_y{i_year+yearbeg}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)   
         else:
            plt.savefig(f'{pathoutput_fig}/map_fromclustering_cluster{n_clusters}_clim_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
         plt.close('')
 
 
      for i_year in range(0,n_years):
         fig = plt.figure(figsize=(6,6))
         ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
         polarCentral_set_latlim((lat_min,90),ax)
         plt.contourf(xgrid,ygrid,cluster_proba_max[i_year,:,:], cmap=segmented_cmap, levels=np.linspace(-0.01,n_clusters+1,n_clusters+2), alpha=1)
         cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
         cb.ax.tick_params(length=0)
         cb.set_ticks(np.arange(0.5,n_clusters+1+0.5,1))
         #cb.set_ticklabels(np.arange(1,n_clusters+1+1,1))
         cb.set_ticklabels(i_cluster_name)
         plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
         plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')   
         #plt.contour(xgrid,ygrid,var_clim_x_y[:,:,0], linestyles='dotted', colors='black')
         ax.add_feature(cfeature.LAND)
         ax.coastlines(linewidth=0.5,color='k')          
         ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
         fig.set_tight_layout(True)
         if clim == 0:
            plt.savefig(f'{pathoutput_fig}/map_fromprobamax_cluster{n_clusters}_distanceEuclidienne_y{i_year+yearbeg}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)   
         else:
            plt.savefig(f'{pathoutput_fig}/map_fromprobamax_cluster{n_clusters}_distanceEuclidienne_clim_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
         plt.close('')

      
      # for contours
      value_for_contour_cluster_firstperiod=np.zeros((n_clusters+1))
      value_for_contour_cluster_currentperiod=np.zeros((n_clusters+1))
      map_proba_nonzero = map_proba.copy() 	
      for i_cluster in range(0,n_clusters+1): 
         value_for_contour_cluster_firstperiod[i_cluster]=round(np.nanmean(np.where(map_proba[0,i_cluster,:,:]>0.1,map_proba[0,i_cluster,:,:],np.nan)),2)    

      for i_year in range(0,n_years):
         for i_cluster in range(0,n_clusters+1):     
            start_color = "FFFFFF"
            end_color = segmented_hex_colors[i_cluster]
            color_map_cluster = create_color_range(start_color, end_color)
            print("color_map_cluster",color_map_cluster)
            
            value_for_contour_cluster_currentperiod[i_cluster]=round(np.nanmean(np.where(map_proba[i_year,i_cluster,:,:]>0.1,map_proba[i_year,i_cluster,:,:],np.nan)),2)            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
            ax.add_feature(cfeature.LAND)
            polarCentral_set_latlim((lat_min,90),ax)
            #plt.title(f'Probability of cluster {i_cluster_name[i_cluster]} of {i_year+yearbeg} - {choix_distance} {end}')
            #plt.contourf(xgrid,ygrid,map_proba[i_year,i_cluster,:,:], levels=np.linspace(0,1,11), cmap=color_map_cluster, alpha=1) # 
            plt.contourf(xgrid,ygrid,map_proba[i_year,i_cluster,:,:], levels=np.linspace(0,1,51), cmap=color_map_cluster, alpha=1) # 
            cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cb.ax.tick_params(length=0)
            cb.ax.tick_params(labelsize=20)
            #plt.clim([0,1])
            plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
            plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')  
            #plt.contour(xgrid,ygrid,var_clim_x_y[:,:,0], linestyles='dotted', colors='black')
            #plt.contour(xgrid,ygrid,map_proba[0,i_cluster,:,:],levels=[np.mean(map_proba[0,i_cluster,:,:])], linestyles='dotted', colors='black')
            #plt.contour(xgrid,ygrid,map_proba[0,i_cluster,:,:],levels=[value_for_contour_cluster_firstperiod[i_cluster]], linestyles='dotted', colors='black',linewidths=1.3)
            #plt.contour(xgrid,ygrid,map_proba[i_year,i_cluster,:,:],levels=[value_for_contour_cluster_currentperiod[i_cluster]], linestyles='dotted', colors='black',linewidths=2.0)
            #ax.scatter(xgrid[200], ygrid[200], marker='o', facecolors='none', edgecolors='green')   #, label='Marker Label'
            ax.add_feature(cfeature.LAND)
            ax.coastlines(linewidth=0.5,color='k')
            ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
            plt.tight_layout()
            
            plt.savefig(f'{pathoutput_fig}/map_proba_cluster{i_cluster}_total{n_clusters}clusters_year{i_year+yearbeg}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}', bbox_inches='tight',dpi=dpi)     
      plt.close('')
      
      if block_years != 0:
         value_for_contour_cluster_firstperiod=np.zeros((n_clusters+1))
         value_for_contour_cluster_currentperiod=np.zeros((n_clusters+1))
         for i_cluster in range(0,n_clusters+1): 
            value_for_contour_cluster_firstperiod[i_cluster]=round(np.nanmean(np.where(map_proba[0:block_years,i_cluster,:,:]>0.1,map_proba[0:block_years,i_cluster,:,:],np.nan)),2)
            #value_for_contour_cluster_firstperiod[i_cluster]=round(np.nanmean(np.where(map_proba[0:block_years,i_cluster,:,:]!=0,map_proba[0:block_years,i_cluster,:,:],np.nan)),2)
         print("value_for_contour_cluster_firstperiod BLOCK",value_for_contour_cluster_firstperiod)
         n_years_block=int(n_years/block_years)
         
         # Figure article 8
         for i_year_block in range(0,n_years_block):
            for i_cluster in range(0,n_clusters+1):
               start_color = "FFFFFF"
               end_color = segmented_hex_colors[i_cluster]
               color_map_cluster = create_color_range(start_color, end_color)
               
               value_for_contour_cluster_currentperiod[i_cluster]=round(np.nanmean(np.where(map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster,:,:]>0.1, map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster,:,:],np.nan)),2)
               mean_map_proba=np.mean(map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster,:,:],axis=0)
               fig = plt.figure(figsize=(6,6))
               ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
               polarCentral_set_latlim((lat_min,90),ax)
               plt.contourf(xgrid,ygrid,mean_map_proba, cmap=color_map_cluster, levels=np.linspace(0,1,150), alpha=1)
               cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)
               cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
               cb.ax.tick_params(length=0)
               cb.ax.tick_params(labelsize=20)
               #plt.clim([0,1])
               cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')
               ax.add_feature(cfeature.LAND)
               ax.coastlines(linewidth=0.5,color='k')
               ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
               plt.tight_layout()
               plt.savefig(f'{pathoutput_fig}/map_proba_cluster{i_cluster}_total{n_clusters}clusters_blockyears{i_year_block}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
               plt.close('')
         
         
         # figure article 9
         for i_year_block in range(0,n_years_block):
            for i_cluster in range(0,n_clusters):
               diff_mean_map_proba=np.mean(map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster+1,:,:],axis=0)-np.mean(map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster,:,:],axis=0)   
                                 
               fig = plt.figure(figsize=(6,6))
               ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
               polarCentral_set_latlim((lat_min,90),ax)
               #plt.title(f'Probability of cluster {i_cluster_name[i_cluster]} of {i_year_block*block_years+yearbeg}-{i_year_block*block_years+block_years+yearbeg} - {choix_distance} {end}')
               #plt.contourf(xgrid,ygrid, diff_mean_map_proba, cmap=cmap_diff, alpha=1, levels=levels) #levels=np.linspace(-0.15,0.15,11)
               if i_cluster == 0:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[0]), (0.8, (1,1,1,1)), (1, segmented_colors[1])]) 
                  levels=np.linspace(-1,0.25,100)          
               elif i_cluster == 1:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[1]), (0.5, (1,1,1,1)), (1, segmented_colors[2])])
                  #levels=np.linspace(-0.08,0.08,100)
                  levels=np.linspace(-1,1,100)
               if i_cluster == 2:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[2]), (0.5, (1,1,1,1)), (1, segmented_colors[3])]) 
                  levels=np.linspace(-1,1,100)    
               plt.contourf(xgrid,ygrid, diff_mean_map_proba, cmap=cmap_diff, levels=levels, alpha=1) #levels=np.linspace(-0.15,0.15,11)
               #plt.contourf(xgrid,ygrid, diff_mean_map_proba, cmap=cmap_diff, alpha=1) #levels=np.linspace(-0.15,0.15,11)
               cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)               
               cb.ax.tick_params(length=0)
               cb.ax.tick_params(labelsize=fontsize)
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')
               ax.add_feature(cfeature.LAND)
               ax.coastlines(linewidth=0.5,color='k')
               ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
               plt.tight_layout()
               plt.savefig(f'{pathoutput_fig}/map_proba_diffcluster{i_cluster+1}-cluster{i_cluster}_total{n_clusters}clusters_blockyears{i_year_block}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
               plt.close('')
      
      
         for i_cluster in range(0,n_clusters):
               diff_mean_map_proba_allyears=np.mean(map_proba[:,i_cluster+1,:,:],axis=0)-np.mean(map_proba[:,i_cluster,:,:],axis=0)   
               
               if i_cluster == 0:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[0]), (0.8, (1,1,1,1)), (1, segmented_colors[1])])
                  levels=np.linspace(-1,0.25,100)
               elif i_cluster == 1:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[1]), (0.5, (1,1,1,1)), (1, segmented_colors[2])])
                  levels=np.linspace(-0.08,0.08,100)
                  levels=np.linspace(-1,1,100)
               elif i_cluster == 2:
                  cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[2]), (0.2, (1,1,1,1)), (1, segmented_colors[3])])
                  levels=np.linspace(-0.25,1,100)
               #cmap_diff = LinearSegmentedColormap.from_list('custom_purple_grey',[(0, segmented_colors[2]), (0.5, (1,1,1,1)), (1, segmented_colors[3])])               
               
               fig = plt.figure(figsize=(6,6))
               ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
               polarCentral_set_latlim((lat_min,90),ax)
               #plt.title(f'Probability of cluster {i_cluster_name[i_cluster]} of {i_year_block*block_years+yearbeg}-{i_year_block*block_years+block_years+yearbeg} - {choix_distance} {end}')
               #plt.contourf(xgrid,ygrid,np.mean(map_proba[i_year_block*block_years:i_year_block*block_years+block_years,i_cluster,:,:],axis=0), levels=np.linspace(0,1,11), cmap=color_map_cluster, alpha=1)
               plt.contourf(xgrid,ygrid, diff_mean_map_proba_allyears, cmap=cmap_diff, alpha=1, levels=levels) # , levels=np.linspace(-0.08,0.08,9) levels=np.linspace(-0.12,0.04,17)  
               cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)

               if i_cluster == 0:
                  cb.set_ticks([-1,-0.75,-0.5,-0.25,0, 0.25])           
               elif i_cluster == 1:
                  #cb.set_ticks([-0.08, -0.04, 0.0, 0.04, 0.08])
                  cb.set_ticks([-1,-0.75,-0.5,-0.25,0, 0.25, 0.5, 0.75, 1])                      
               elif i_cluster == 2:
                  cb.set_ticks([-0.25,0,0.25, 0.5, 0.75, 1])

               cb.ax.tick_params(length=0)
               cb.ax.tick_params(labelsize=fontsize)
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
               plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')
               ax.add_feature(cfeature.LAND)
               ax.coastlines(linewidth=0.5,color='k')
               ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
               plt.tight_layout()
               plt.savefig(f'{pathoutput_fig}/map_proba_diffcluster{i_cluster+1}-cluster{i_cluster}_allyears_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
               plt.close('')         

   if plot_ts == 1:
      print("plot_ts")
      ts_proba_clusterwith0_year=np.zeros((n_years, n_clusters+1))
      for i_cluster in range(0,n_clusters+1):
         for i_year in range(0,n_years):
            ts_proba_clusterwith0_year[i_year,i_cluster] = np.nansum(map_proba[i_year,i_cluster,:,:])/np.nansum(map_proba[i_year,:,:,:])
            
      
      ts_probaarea_cluster_year=np.zeros((n_years, n_clusters+1))
      ts_probaarea_cluster_year_notnormalized=np.zeros((n_years, n_clusters+1))
      true_cluster=np.zeros((var.shape[1], var.shape[2]))
      for i_cluster in range(0,n_clusters+1):
         for i_year in range(0,n_years):
            ts_probaarea_cluster_year[i_year,i_cluster] = (np.nansum(map_proba[i_year,i_cluster,:,:]*cell_area[:,:]))/(np.nansum(map_proba[i_year,:,:,:]*cell_area[:,:]))
            ts_probaarea_cluster_year_notnormalized[i_year,i_cluster] = np.nansum(map_proba[i_year,i_cluster,:,:]*cell_area[:,:])
      
      # ts proba trend - Figure article 7  
      plt.figure(figsize=(8,4))
      for i_cluster in range(0,n_clusters+1):
         print("Cluster", i_cluster)
         model=np.polyfit(year_range,ts_probaarea_cluster_year[:,i_cluster],1)
         predict = np.poly1d(model)
         y_reg = predict(year_range)
         plt.plot(year_range,y_reg*100, c=segmented_cmap(i_cluster), linestyle='--', alpha=1, linewidth=1)
         plt.plot(year_range,ts_probaarea_cluster_year[:,i_cluster]*100, marker_cluster[i_cluster], c=segmented_cmap(i_cluster), alpha=1, linewidth=2, label=f'{i_cluster_name[i_cluster]} ({round(model[0]*1000,1)} % per decade)')
         slope, intercept, r_value, p_value, std_err = linregress(year_range[:],ts_probaarea_cluster_year[:,i_cluster])
         # Display the calculated values
         print(f"Slope probaxarea {i_cluster}: {slope}")
         print(f"Standard Error probaxarea {i_cluster}: {std_err}")
         print(f"P-value probaxarea {i_cluster}: {p_value}")
         print(f"np.mean(y_reg*100) probaxarea {i_cluster}",np.mean(y_reg*100))
      x_test = np.array([1, 2, 3, 4, 5])
      y_test = np.array([2, 2.8, 2, 1.8, 2.8])
      slope, intercept, r_value, p_value, std_err = linregress(x_test,y_test)
  
      plt.grid(linestyle = '--', linewidth = 0.5)
      #plt.ylabel(f'Probability x Area / total area (%) - {choix_distance}')
      plt.ylabel(f'Total Probability (%)', fontsize=fontsize)
      plt.xticks(np.arange(yearbeg,yearbeg + n_years,4), fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      plt.ticklabel_format(useOffset=False)
      #ordered_handles = [handles[idx] for idx in order_cluster_name]
      #ordered_labels = [labels[idx] for idx in order_cluster_name]
      #plt.legend(ordered_handles, ordered_labels, loc=(0.78, 0.44), fontsize=fontsize)
      handles, labels = plt.gca().get_legend_handles_labels()
      plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], fontsize=12) #loc=(0.6, 0.51)
      plt.margins(x=0.01)
      plt.ylim(8, 60)
      plt.grid('both')
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/ts_probaxarea_normalized_trend_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}')
      plt.close('') 
        
      
      ts_cell_area_cluster_max=np.zeros((n_years, n_clusters+1))
      ts_cell_area_cluster_max_relatif=np.zeros((n_years, n_clusters+1))
      for i_year in range(0,n_years):
         for i_cluster in range(0,n_clusters+1):
            ts_cell_area_cluster_max[i_year,i_cluster]=np.nansum(mask_cluster_max[i_year,i_cluster,:,:]*cell_area[:,:])
            ts_cell_area_cluster_max_relatif[i_year,i_cluster]=ts_cell_area_cluster_max[i_year,i_cluster]/ts_cell_area_cluster_max[0,i_cluster]
                     
      # figure article 6a	
      plt.figure(figsize=(8,4))
      for i_cluster in range(0,n_clusters+1):
         model=np.polyfit(year_range,ts_cell_area_cluster_max[:,i_cluster],1)
         predict = np.poly1d(model)
         y_reg = predict(year_range)
         slope, intercept, r_value, p_value, std_err = linregress(year_range[:],ts_cell_area_cluster_max[:,i_cluster])
         # Display the calculated values
         print(f"Slope cluster {i_cluster}: {slope}")
         print(f"Standard Error cluster {i_cluster}: {std_err}")
         print(f"P-value cluster {i_cluster}: {p_value}")
         plt.plot(year_range, y_reg/10**12, c=segmented_cmap(i_cluster), linestyle='--', alpha=1, linewidth=1)
         plt.plot(year_range,ts_cell_area_cluster_max[:,i_cluster]/10**12, marker_cluster[i_cluster], c=segmented_cmap(i_cluster), alpha=1, linewidth=2, label=f'{i_cluster_name[i_cluster]} ({round(model[0]*10/10**12,1)}.10⁶ km² per decade)')
      plt.grid(linestyle = '--', linewidth = 0.5)
      plt.ylabel(f'Total area (10⁶ km²)', fontsize=fontsize)
      plt.xticks(np.arange(yearbeg,yearbeg + n_years,4),fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      plt.ticklabel_format(useOffset=False)
      plt.margins(x=0.01)
      plt.margins(y=0.05)
      plt.ylim(1.7, 12.7)
      handles, labels = plt.gca().get_legend_handles_labels()
      plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], fontsize=12, ncol=1) #(0.74,0.62), , loc=(0.82,0.63)
      plt.grid('both')
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/ts_aire_probamax_trend_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}')
      plt.close('') 
      
      # MIZ
      MIZ_openocean_map=np.zeros((n_years,var.shape[1], var.shape[2]))
      MIZ_MIZ_map=np.zeros((n_years,var.shape[1], var.shape[2]))
      MIZ_pack_map=np.zeros((n_years,var.shape[1], var.shape[2]))
      MIZ_area=np.zeros((3,n_years), dtype=int)
      var_year_x_y=np.zeros((n_years,var.shape[1], var.shape[2]))
      for x in range(var.shape[2]):
         for y in range(var.shape[1]):
            if var_land[y,x] == 0:
               for i_year in range(0,n_years):
                  var_year_x_y[i_year,y,x]=np.nanmean(var_with_0[nb_time_step*i_year:nb_time_step*i_year+nb_time_step,y,x], axis=0)  
                  if (var_year_x_y[i_year,y,x] >= 0. and var_year_x_y[i_year,y,x]<0.15) :
                     MIZ_openocean_map[i_year,y,x]=1
                     MIZ_area[0,i_year] += cell_area[y,x]
                  elif var_year_x_y[i_year,y,x]>=0.15 and var_year_x_y[i_year,y,x]<=0.8:
                     MIZ_MIZ_map[i_year,y,x]=1
                     MIZ_area[1,i_year] += cell_area[y,x]
                  elif var_year_x_y[i_year,y,x]>0.8 and var_year_x_y[i_year,y,x]<=1. :
                     MIZ_pack_map[i_year,y,x]=1
                     MIZ_area[2,i_year] += cell_area[y,x]
      

      i_cluster_name_MIZ=["Open-ocean", "MIZ", "Packed ice"]
      color_MIZ = (["blue", "green", "orange"])
      marker_MIZ = (['o-', 's-', 'd-'])
      
      # figure article 6b
      plt.figure(figsize=(8,4))
      for i_cluster in range(0,3,1):
         model=np.polyfit(year_range,MIZ_area[i_cluster,:],1)
         predict = np.poly1d(model)
         y_reg = predict(year_range)
         slope, intercept, r_value, p_value, std_err = linregress(year_range[:],MIZ_area[i_cluster,:])
         # Display the calculated values
         print(f"Slope MIZ {i_cluster}: {slope}")
         print(f"Standard Error MIZ {i_cluster}: {std_err}")
         print(f"P-value MIZ {i_cluster}: {p_value}")
         plt.plot(year_range, y_reg/10**12, color=color_MIZ[i_cluster], linestyle='--', alpha=1, linewidth=1)
         plt.plot(year_range, MIZ_area[i_cluster,:]/10**12, marker_MIZ[i_cluster], color=color_MIZ[i_cluster], linewidth=2, label=f'{i_cluster_name_MIZ[i_cluster]} ({round(model[0]*10/10**12,1)} 10⁶km² per decade)')
      plt.grid(linestyle = '--', linewidth = 0.5)
      plt.ylabel(f'Total area (10⁶ km²)', fontsize=fontsize)
      plt.xticks(np.arange(yearbeg,yearbeg + n_years,4), fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      plt.margins(x=0.01)
      plt.margins(y=0.05)
      plt.ticklabel_format(useOffset=False)
      plt.grid('both')
      #plt.legend(loc=(0.01, 0.18), fontsize=fontsize)
      #plt.ylim(1.7, 12.7)
      plt.legend(fontsize=12)
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/ts_MIZarea_trend_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}')
      plt.close('') 
   
   ### REGIME ###
   if plot_regime == 1:
      print("plot_regime == 1")
         
      ts_proba_gridcell2_clusterwith0_year=np.zeros((n_years, n_clusters+1))
      ts_proba_gridcell5_clusterwith0_year=np.zeros((n_years, n_clusters+1))
      for i_cluster in range(0,n_clusters+1):
         for i_year in range(0,n_years):
            ts_proba_gridcell2_clusterwith0_year[i_year,i_cluster] = map_proba[i_year,i_cluster,190,105]/np.nansum(map_proba[i_year,:,190,105])*100
            ts_proba_gridcell5_clusterwith0_year[i_year,i_cluster] = map_proba[i_year,i_cluster,230,226]/np.nansum(map_proba[i_year,:,230,226])*100
            
           
      # Figure article 10a
      fig = plt.figure(figsize=(8,4))
      #fig.patch.set_facecolor('orange')
      #fig.patch.set_alpha(0.2)
      for i_cluster in range(0,n_clusters+1):
         plt.plot(year_range,ts_proba_gridcell2_clusterwith0_year[:,i_cluster], marker_cluster[i_cluster], c=segmented_cmap(i_cluster), alpha=1, linewidth=2, label=f'{i_cluster_name[i_cluster]}') 
      plt.grid(linestyle = '--', linewidth = 0.5)
      plt.ylabel(f'Probability of the "star" cell (%)', fontsize=fontsize)
      plt.xticks(np.arange(yearbeg,yearbeg + n_years,4), fontsize=fontsize)
      plt.yticks(fontsize=fontsize) 
      #plt.tick_params(labelbottom=False)
      plt.margins(x=0.01)
      plt.margins(y=0.05)
      plt.ticklabel_format(useOffset=False)
      #handles, labels = plt.gca().get_legend_handles_labels()
      #plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], fontsize=fontsize)
      plt.grid('both')
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/ts_proba_1gridcell2_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}')
      plt.close('') 
        
      #Figure article 10b
      fig = plt.figure(figsize=(8,4))
      #fig.patch.set_facecolor('crimson')
      #fig.patch.set_alpha(0.2)
      for i_cluster in range(0,n_clusters+1):
         plt.plot(year_range,ts_proba_gridcell5_clusterwith0_year[:,i_cluster], marker_cluster[i_cluster], c=segmented_cmap(i_cluster), alpha=1, linewidth=2) #, label=f'{i_cluster_name[i_cluster]}'
      plt.grid(linestyle = '--', linewidth = 0.5)
      plt.ylabel(f'Probability of the "triangle" cell (%)',fontsize=fontsize)
      plt.xticks(np.arange(yearbeg,yearbeg + n_years,4),fontsize=fontsize)
      #plt.tick_params(labelbottom=False)
      plt.yticks(fontsize=fontsize) 
      plt.margins(x=0.01)
      plt.margins(y=0.05)
      plt.ticklabel_format(useOffset=False)
      #handles, labels = plt.gca().get_legend_handles_labels()
      #plt.legend([handles[idx] for idx in order_cluster_name], [labels[idx] for idx in order_cluster_name], fontsize=fontsize)
      plt.grid('both')
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/ts_proba_1gridcell5_cluster{n_clusters}_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}')
      plt.close('') 

      year_beg_last_abrupt_change = np.zeros((var.shape[1], var.shape[2]), dtype=int)
      year_end_last_abrupt_change = np.zeros((var.shape[1], var.shape[2]), dtype=int)
      mask_cluster_max=np.zeros((n_years,n_clusters+1,var.shape[1], var.shape[2]))
      regimes_class=np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
               
      for x in range(var.shape[2]):
         for y in range(var.shape[1]):
            if var_land[y,x] == 0:
               year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x]=find_last_abrupt_change(cluster_proba_max[:,y,x], min_length, tolerance, yearbeg)

               if year_beg_last_abrupt_change[y,x] == yearbeg and year_end_last_abrupt_change[y,x] == yearend:
               #if year_beg_last_abrupt_change[y,x] <= yearbeg and year_end_last_abrupt_change[y,x] >= yearend:
                  #print("x,y, yearbeg,yearend stable regime over the whole period 1", x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x])
                  regimes_class[y,x]=1
               elif year_beg_last_abrupt_change[y,x] > yearbeg and year_end_last_abrupt_change[y,x] == yearend:
                  #print("x,y, yearber,yearend abrupt change toward a stable regime 3", x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x])
                  regimes_class[y,x]=2
               elif year_beg_last_abrupt_change[y,x] == -1 and year_beg_last_abrupt_change[y,x] == -1:
                  #print("x,y, yearbeg,yearend no stable regime over the whole period 2", x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x])
                  regimes_class[y,x]=3                  
               elif year_beg_last_abrupt_change[y,x] == yearbeg and year_end_last_abrupt_change[y,x] <yearend:
                  #print("x,y, yearbeg,yearend abrupt change toward no stable regime 4", x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x])
                  regimes_class[y,x]=4
               elif year_beg_last_abrupt_change[y,x] > yearbeg and year_end_last_abrupt_change[y,x] <yearend:
                  #print("x,y, yearbeg,yearend stable regime between unstable regimes 5", x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x])
                  #regimes_class[y,x]=5
                  regimes_class[y,x]=2
               else:
                  print("ATTENTION REGIME FOUR x,y, yearbeg,yearend 5",x , y, year_beg_last_abrupt_change[y,x], year_end_last_abrupt_change[y,x]) 
               for i_cluster in range(0,n_clusters+1): 
                  mask_cluster_max[i_year,i_cluster,y,x]=np.where(cluster_proba_max[i_year,y,x]==i_cluster,1,0)
                   
             
      #regime_name=['stable regime \n over the \n whole period', 'no stable regime \n over the \n whole period', 'abrupt change \n toward a \n stable regime', 'abrupt change \n toward  \n no stable regime', 'stable regime(s) \n between \n unstable regimes']
      regime_name=['Stable', 'Stabilization', 'Unstable', 'Destabilization' ]
      
      
      set3_cmap = plt.cm.get_cmap('Set3', 5)
      # Extract and display the colors, excluding color 1
      colors = [set3_cmap(i) for i in range(5) if i != 2]
      # Create a custom colormap with the remaining colors
      custom_set3 = plt.cm.colors.ListedColormap(colors)
      
      var_reshaped = var_with_0.reshape(n_years, nb_time_step, var.shape[1], var.shape[2])
      # Take mean over each 73-step block (axis=1)
      var_yearly_mean = var_reshaped.mean(axis=1)  # shape: (45, 448, 304)
      
      dominant_cluster = np.argmax(map_proba, axis=1)
      # tst equivalent to year_map_nan
      print("np.shape(dominant_cluster)", np.shape(dominant_cluster))
      #print("dominant_cluster", dominant_cluster)
      
      
      from matplotlib import gridspec

      # Simple data to display in various forms
      y0 = var_yearly_mean[:,287,220]
      c_dom_0 = dominant_cluster[:,287,220]

      
      y1 = var_yearly_mean[:,210,153]
      c_dom_1 = dominant_cluster[:,210,153]
      
      y2 = var_yearly_mean[:,170,60]
      c_dom_2 = dominant_cluster[:,170,60]
      
      y3 = var_yearly_mean[:,200,192]
      c_dom_3 = dominant_cluster[:,200,192]
      
      y4 = var_yearly_mean[:,190,105]
      c_dom_4 = dominant_cluster[:,190,105]
      
      y5 = var_yearly_mean[:,230,226]
      c_dom_5 = dominant_cluster[:,230,226]

      
      #190,105
      print("var_yearly_mean[:,170,60]", var_yearly_mean[:,170,60])
      print("dominant_cluster[:,170,60]", dominant_cluster[:,170,60])
            

      # Map cluster index to color
      colors_mapped_0 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_0])
      colors_mapped_1 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_1])
      colors_mapped_2 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_2])
      colors_mapped_3 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_3])
      colors_mapped_4 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_4])
      colors_mapped_5 = np.array([segmented_colors[int(c) % len(segmented_colors)] for c in c_dom_5])
      
      marker_mapped_0 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_0])
      marker_mapped_1 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_1])
      marker_mapped_2 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_2])
      marker_mapped_3 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_3])
      marker_mapped_4 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_4])
      marker_mapped_5 = np.array([marker_cluster_2[int(c) % len(marker_cluster_2)] for c in c_dom_5])
      print("marker_mapped_0",marker_mapped_0)

      
      #figure article 11
      nb_regime=4
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      #plt.title(f'Regime of Arctic sea-ice seasonal cycle - {choix_distance} {end}')
      plt.contourf(xgrid,ygrid,regimes_class, levels=np.linspace(0.01,nb_regime+0.01,nb_regime+1), cmap=custom_set3, alpha=0.7) #plt.cm.get_cmap('Set3', 5)
      cb = plt.colorbar(ax=ax, orientation="horizontal", fraction=0.06, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.set_ticks(np.arange(0.5,nb_regime+0.5,1))
      cb.set_ticklabels(regime_name, fontsize=fontsize)
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')
      plt.tight_layout()      
      ax.scatter(xgrid[226], ygrid[230], marker='^', s=50, facecolors='none', edgecolors='black')
      ax.scatter(xgrid[105], ygrid[190], marker='*', s=50, facecolors='none', edgecolors='black')
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      plt.savefig(f'{pathoutput_fig}/map_regimes_class_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi) 
      plt.close('')
      
      year_beg_stabilization = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      year_beg_destabilization = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      
      year_beg_stabilization=np.where(regimes_class == 2,year_beg_last_abrupt_change,np.nan)
      year_beg_destabilization=np.where(regimes_class == 4,year_end_last_abrupt_change,np.nan)
      
      #figure article 12a
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      #plt.title(f'First year of Stabilization - {choix_distance} {end}')
      #plt.title(f'First year of Stabilization')
      plt.contourf(xgrid,ygrid,year_beg_stabilization, levels=np.arange(1978,2023), cmap='viridis', alpha=0.5) # levels=np.linspace(0,1,11),
      ax.scatter(xgrid[105], ygrid[190], marker='*', s=50, facecolors='none', edgecolors='black')
      ax.scatter(xgrid[226], ygrid[230], marker='^', s=50, facecolors='none', edgecolors='black')
      cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.set_ticks(np.arange(yearbeg, yearend,4))
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/map_yearbeg_stabilization_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi) 
      plt.close('')
      
      #figure article 12b
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      #plt.title(f'First year of Destabilization - {choix_distance} {end}')
      #plt.title(f'First year of Destabilization')
      plt.contourf(xgrid,ygrid,year_beg_destabilization, levels=np.arange(1978,2023), cmap='viridis', alpha=0.5) # levels=np.linspace(0,1,11),
      ax.scatter(xgrid[105], ygrid[190], marker='*', s=50, facecolors='none', edgecolors='black')
      ax.scatter(xgrid[226], ygrid[230], marker='^', s=50, facecolors='none', edgecolors='black')
      cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.set_ticks(np.arange(yearbeg, yearend,4))
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      plt.tight_layout()
      plt.savefig(f'{pathoutput_fig}/map_yearbeg_destabilization_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi) 
      plt.close('')
      
      
      most_common_number_stabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      most_common_number_stabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      most_common_number_destabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      most_common_number_destabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      
      occurrences_stabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      occurrences_stabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      occurrences_destabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      occurrences_destabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      
      percentage_stabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      percentage_stabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      percentage_destabilization_stableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      percentage_destabilization_unstableregime = np.ones((var.shape[1], var.shape[2]), dtype=int)*np.nan
      
      for x in range(var.shape[2]):
         for y in range(var.shape[1]):
            if regimes_class[y,x] == 2: # stabilization
               most_common_number_stabilization_stableregime[y,x], occurrences_stabilization_stableregime[y,x], percentage_stabilization_stableregime[y,x] = find_most_common_number(cluster_proba_max[(int(year_beg_stabilization[y,x])-yearbeg):,y,x])
               most_common_number_stabilization_unstableregime[y,x], occurrences_stabilization_unstableregime[y,x], percentage_stabilization_unstableregime[y,x] =find_most_common_number(cluster_proba_max[0:(int(year_beg_stabilization[y,x])-yearbeg),y,x])
            if regimes_class[y,x] == 4:  # destabilization
               most_common_number_destabilization_stableregime[y,x], occurrences_destabilization_stableregime[y,x], percentage_destabilization_stableregime[y,x] =find_most_common_number(cluster_proba_max[0:(int(year_beg_destabilization[y,x])-yearbeg),y,x])
               most_common_number_destabilization_unstableregime[y,x], occurrences_destabilization_unstableregime[y,x], percentage_destabilization_unstableregime[y,x] =find_most_common_number(cluster_proba_max[(int(year_beg_destabilization[y,x])-yearbeg):,y,x])
                   
      i_cluster_name=["Open\nocean\ncluster", "Partial\nwinter\nfreezing\ncluster", "Full\nwinter\nfreezing\ncluster", "Permanent\nsea-ice\ncluster"]
      
      #figure article 12c
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      plt.contourf(xgrid,ygrid,most_common_number_stabilization_stableregime, cmap=segmented_cmap, levels=np.linspace(-0.01,n_clusters+1,n_clusters+2), alpha=1)
      cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.set_ticks(np.arange(0.5,n_clusters+1+0.5,1))
      cb.set_ticklabels(i_cluster_name)
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')   
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')
      ax.scatter(xgrid[105], ygrid[190], marker='*', s=50, facecolors='none', edgecolors='black')
      ax.scatter(xgrid[226], ygrid[230], marker='^', s=50, facecolors='none', edgecolors='black')          
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      fig.set_tight_layout(True)
      plt.savefig(f'{pathoutput_fig}/map_most_common_number_stabilization_stableregime_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
      plt.close('')
      
     
      
      #figure article 12d
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=315.0))
      polarCentral_set_latlim((lat_min,90),ax)
      plt.contourf(xgrid,ygrid,most_common_number_destabilization_stableregime, cmap=segmented_cmap, levels=np.linspace(-0.01,n_clusters+1,n_clusters+2), alpha=1)
      cb = plt.colorbar(ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
      cb.ax.tick_params(length=0)
      cb.set_ticks(np.arange(0.5,n_clusters+1+0.5,1))
      cb.set_ticklabels(i_cluster_name)
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.15], linestyles='dotted', linewidths=1.3, colors='black')
      plt.contour(xgrid,ygrid,np.mean(var_clim_x_y[:,:,:],axis=2), levels=[0.80], linestyles='dotted', linewidths=2, colors='black')
      ax.scatter(xgrid[105], ygrid[190], marker='*', s=50, facecolors='none', edgecolors='black')
      ax.scatter(xgrid[226], ygrid[230], marker='^', s=50, facecolors='none', edgecolors='black')    
      ax.add_feature(cfeature.LAND)
      ax.coastlines(linewidth=0.5,color='k')          
      ax.gridlines(color='C7',lw=1,ls=':',draw_labels=False,rotate_labels=False,ylocs=[60,70,80])
      fig.set_tight_layout(True)
      plt.savefig(f'{pathoutput_fig}/map_most_common_number_destabilization_stableregime_total{n_clusters}clusters_{yearbeg}{yearend}{time_step}_{choix_distance}{end}.{format_fig}', format=f'{format_fig}',dpi=dpi)     
      plt.close('')
      
               

            
      
      
END = datetime.now()
DIFF = (END - START)
print(f'It took {int(DIFF.total_seconds() / 60)} minutes or {int(DIFF.total_seconds() / 3600)} hours or {int(DIFF.total_seconds() / 86400)} days' )
