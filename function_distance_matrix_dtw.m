__author__ = 'Saeid SOHILY-KHAH'
%Calculate matrix of dtw distances between all time series pairs with Sakoe-Chiba Band 
%Inputs: 
%       - sakoe_chiba_band (dynamic time warping sc)
%Output:
%       - distance_matrix_dtw
%-------------------------------------------------------------------------------------------
clear; clc;
sakoe_chiba_band = Inf;

%set dataSet and its variable
%-------------------------------------------------------------------------------------------
[dataSet] = function_read_dataset();
n = size(dataSet,1); %number of time series
m = size(dataSet,2); %length of time series

%compute DTW distance between all pairs
%-------------------------------------------------------------------------------------------
for i = 1 : n
	for j = i : n
		distance_matrix_dtw(i,j) = function_dtw(dataSet(i,:)',dataSet(j,:)',sakoe_chiba_band);
		distance_matrix_dtw(j,i) = distance_matrix_dtw(i,j);
	end;
end;

%save DTW distance between all pairs
%-------------------------------------------------------------------------------------------
save('distance_matrix_dtw.txt','distance_matrix_dtw','-ascii');
