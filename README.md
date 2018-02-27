# Weighted Dynamic Time Warping (WDTW)

Searching the best alignment that matches two time series is an important task for many researcher. The Dynamic Time Warping (DTW) is among the most frequently used metrics for time series in several domains as signal processing, data mining or machine learning. It finds the optimal alignment between two time series, and captures flexible similarities by aligning the elements inside both sequences. Intuitively, the time series are warped non-linearly in the time dimension to match each other.

Dynamic time warping (DTW) is currently a well-known dissimilarity measure on time series and sequences, since it makes them possible to capture temporal distortions. The Dynamic Time Warping (DTW) between time series x_i and time series x_j , with the aim of minimization of the mapping cost, is defined by:

                                            DTW(x_i , x_j) = min φ(x_it′ , x_jt)
                                                             π∈A
               
where A is the set of all alignments possible between two time series, and φ : R × R → R + is a positive, real-valued, divergence function (generally Euclidean norm).              

For measures based on the time warp alignments, the integration of a weighting vector allows one to differently weigh the different time stamps of the series under consideration. The role of the weight vector is to indicate the importance of each time stamp.

The Weighted Dynamic Time Warping (WDTW) between the time series x and the weighted time series (c, w) is defined by:

                                            WDTW(x , (c,w)) = min f(w_t) φ(x_t′ , c_t)
                                                              π∈A
               
where A is the set of all alignments possible between two time series, f : (0, 1] → R+ is a non-increasing function and φ : R × R → R + is a positive, real-valued, dissimilarity function. 
The above definition thus generalizes the DTW to the case where the different instants are weighted.

References: 
R.Bellman and S. Dreyfus. “Applied Dynamic Programming.” In: New Jersey: Princeton Univ. Press (1962)

F. Itakura. “Minimum prediction residual principle applied to speech recognition.” In: Acoustics, Speech and Signal Processing, IEEE Transactions on 23.1 (1975), pp. 67–72

J.B Kruskall and M. Liberman. The symmetric time warping algorithm: From continuous to discrete. In Time Warps, String Edits and Macromolecules. Addison-Wesley., 1983

Saeid Soheily-Khah, Ahlame Douzal-Chouakria, and Eric Gaussier. “Generalized k-means-based clustering for temporal data under weighted and kernel time warp.” In: Journal of Pattern Recognition Letters (2016)
