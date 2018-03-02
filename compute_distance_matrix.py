__author__ = 'Saeid SOHILY-KHAH'
"""
Compute (w)dtw distance matrix for a UCR time series dataset
Note:
    for wdtw, the weighting function is considered as: f(w) = w^-(alpha) [line 86],
    alpha = 1 [line 137], 
    and w is considered as a random weight vector. [line 178] 
"""
import time
import numpy as np
import matplotlib.pylab as plt


def dtw(x, y, sakoe_chiba_band):
    # Compute the size of each time serie
    # -------------------------------------------------------------------------------------------
    n = len(x)
    m = len(y)

    # Local Cost Matrix (Dissimilarities)
    # -------------------------------------------------------------------------------------------
    dist = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            dist[i][j]  = (x[i]-y[j])**2

    # Cost Matrix with Sakoe-Chiba Band
    # -------------------------------------------------------------------------------------------
    dtw_cost = np.empty((n, m))
    dtw_cost.fill(0)
    dtw_cost[0][0] = dist[0][0]
    for i in range(1,n):
        dtw_cost[i][0] = dtw_cost[i-1][0] + dist[i][0]
    for j in range(1,m):
        dtw_cost[0][j] = dtw_cost[0][j-1] + dist[0][j]
    for i in range(1,n):
        for j in range(1,m):
            if abs(i-j) <= sakoe_chiba_band:
                choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
                dtw_cost[i][j] = dist[i][j] + min(choices)
            else:
                dtw_cost[i][j] = float('inf')

    # Compute Warping Path
    # -------------------------------------------------------------------------------------------
    i = n-1
    j = m-1
    path = np.empty((n, m))
    path.fill(0)
    path[n-1][m-1] = 1
    size_warping_path = 1
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
            if dtw_cost[i-1,j-1] == min(choices):
                i = i - 1
                j = j - 1
            elif dtw_cost[i,j-1] == min(choices):
                j = j - 1
            else:
                i = i - 1
        path[i][j] = 1
        size_warping_path += size_warping_path

    # Return Dynamic Time Warping Distance
    # -------------------------------------------------------------------------------------------
    return dtw_cost[-1][-1]


def wdtw(x, y, weight_vector, sakoe_chiba_band):
    global alpha
    # Compute the size of each time serie
    # -------------------------------------------------------------------------------------------
    n = len(x)
    m = len(y)

    # Local Cost Matrix (Dissimilarities)
    # -------------------------------------------------------------------------------------------
    dist = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            dist[i][j]  = ( 1 / (pow(weight_vector[j],alpha)) ) * (x[i]-y[j])**2

    # Cost Matrix with Sakoe-Chiba Band
    # -------------------------------------------------------------------------------------------
    dtw_cost = np.empty((n, m))
    dtw_cost.fill(0)
    dtw_cost[0][0] = dist[0][0]
    for i in range(1,n):
        dtw_cost[i][0] = dtw_cost[i-1][0] + dist[i][0]
    for j in range(1,m):
        dtw_cost[0][j] = dtw_cost[0][j-1] + dist[0][j]
    for i in range(1,n):
        for j in range(1,m):
            if abs(i-j) <= sakoe_chiba_band:
                choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
                dtw_cost[i][j] = dist[i][j] + min(choices)
            else:
                dtw_cost[i][j] = float('inf')

    # Compute Warping Path
    # -------------------------------------------------------------------------------------------
    i = n-1
    j = m-1
    path = np.empty((n, m))
    path.fill(0)
    path[n-1][m-1] = 1
    size_warping_path = 1
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            choices = dtw_cost[i-1][j], dtw_cost[i][j-1], dtw_cost[i-1][j-1]
            if dtw_cost[i-1,j-1] == min(choices):
                i = i - 1
                j = j - 1
            elif dtw_cost[i,j-1] == min(choices):
                j = j - 1
            else:
                i = i - 1
        path[i][j] = 1
        size_warping_path += size_warping_path

    # Return Weighted Dynamic Time Warping Distance
    # -------------------------------------------------------------------------------------------
    return dtw_cost[-1][-1]


if __name__ == '__main__':
    sakoe_chiba_band = float('inf') #
    alpha = 1 # parameter of non-increasing weight function (see https://www.researchgate.net/publication/298789494_Generalized_k-means-based_clustering_for_temporal_data_under_weighted_and_kernel_time_warp)

    # Read data from file
    # -------------------------------------------------------------------------------------------

    dataSet = np.loadtxt('dataSet.txt')
    initalClusters = dataSet[:,0]
    dataSet = dataSet[:,1:]

    (N,l) = dataSet.shape # set N as number of time series and l the length of time series
    print("number of time series: %i " %N)
    print("length of time series: %i " %l)

    # Start timer to compute runtime
    # -------------------------------------------------------------------------------------------
    start_time = time.time()

    # Compute DTW distance between all pairs in dataset
    # -------------------------------------------------------------------------------------------
    print('computing dtw distance matrix...')
    distance_matrix_dtw = np.empty((N, N)) # N: number of time series in dataset
    distance_matrix_dtw.fill(0)
    for i in range(N):
        for j in range(i,N):
            distance_matrix_dtw[i][j] = dtw(dataSet[i,:],dataSet[j,:],sakoe_chiba_band)
            distance_matrix_dtw[j][i] = distance_matrix_dtw[i][j]

    np.savetxt('distance_matrix_dtw.txt', distance_matrix_dtw)

    # Print runtime
    # -------------------------------------------------------------------------------------------
    print("dtw runtime is equal to: %s seconds" % round((time.time() - start_time), 2))

    # Start timer to compute runtime
    # -------------------------------------------------------------------------------------------
    start_time = time.time()

    # Compute WDTW distance between all pairs in dataset
    # -------------------------------------------------------------------------------------------
    print('computing wdtw distance matrix...')
    # weight_vector = np.ones(l)/l # uniform weighting
    weight_vector = np.random.rand(l)
    weight_vector = weight_vector / np.sum(weight_vector) # random weighting
    distance_matrix_wdtw = np.empty((N, N)) # N: number of time series in dataset
    distance_matrix_wdtw.fill(0)
    for i in range(N):
        for j in range(i,N):
            distance_matrix_wdtw[i][j] = wdtw(dataSet[i,:], dataSet[j,:], weight_vector, sakoe_chiba_band)
            distance_matrix_wdtw[j][i] = distance_matrix_wdtw[i][j]

    np.savetxt('distance_matrix_wdtw.txt', distance_matrix_wdtw)

    # Print runtime
    # -------------------------------------------------------------------------------------------
    print("wdtw runtime is equal to: %s seconds" % round((time.time() - start_time), 2))

    # Plot the dtw/wdtw distance matrix
    # -------------------------------------------------------------------------------------------
    plt.subplot(121)
    distance_matrix_dtw = distance_matrix_dtw / np.max(distance_matrix_dtw) # normalize matrix
    plt.imshow(distance_matrix_dtw, interpolation='nearest', cmap=plt.cm.gnuplot2, extent=(0.5,N+0.5,0.5,N+0.5))
    plt.colorbar()
    plt.title('normalized distance matrix dtw')

    plt.subplot(122)
    distance_matrix_wdtw = distance_matrix_wdtw / np.max(distance_matrix_wdtw) # normalize matrix
    plt.imshow(distance_matrix_wdtw, interpolation='nearest', cmap=plt.cm.gnuplot2, extent=(0.5,N+0.5,0.5,N+0.5))
    plt.colorbar()
    plt.title('normalized distance matrix wdtw')

    plt.show()
