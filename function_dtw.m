__author__ = 'Saeid SOHILY-KHAH'
%Compute Dynamic Time Warping distance between two time series X and Y with Sakoe-Chiba Band 
%-------------------------------------------------------------------------------------------
function [size_warping_path, path, dtw_distance] = function_dtw(X, Y, sakoe_chiba_band)

%Compute the size of each time serie
%-------------------------------------------------------------------------------------------
n = size(X,1);
m = size(Y,1);

%Local Cost Matrix (Dissimilarities)
%-------------------------------------------------------------------------------------------
C = (repmat(X(:),1,n)-repmat(Y(:)',m,1)).^2; 

%Cost Matrix with Sakoe-Chiba Band
%-------------------------------------------------------------------------------------------
dtw = zeros(n,m);
dtw(1,1) = C(1,1);

for i = 2 : n
	dtw(i,1) = dtw(i-1,1) + C(i,1);
end;
for j = 2 : m
	dtw(1,j) = dtw(1,j-1) + C(1,j);
end;
for i = 2 : n
	for j = 2 : m
		if abs(i-j) <= sakoe_chiba_band 
			dtw(i,j) = C(i,j) + min(dtw(i-1,j-1) , min(dtw(i-1,j),dtw(i,j-1)));
		else
			dtw(i,j) = Inf;
		end;
		
	end;
end;

%Compute Warping Path
%-------------------------------------------------------------------------------------------
i = n;
j = m;
path = zeros(n,m);
path(n,m) = 1;
size_warping_path = 1;

while (i > 1) || (j > 1)
	if (i == 1)
		j = j-1;
	elseif (j == 1)
		i = i-1;
	else
		if (dtw(i-1,j-1) == min(dtw(i-1,j-1) , min(dtw(i-1,j),dtw(i,j-1))))
			i = i-1;
			j = j-1;
		elseif (dtw(i,j-1) == min(dtw(i-1,j-1) , min(dtw(i-1,j),dtw(i,j-1))))
			j = j-1;
		else
			i = i-1;
		end;
	end;
	path(i,j) = 1;
	size_warping_path = size_warping_path + 1;
end;

%Compute Dynamic Time Warping Distance
%-------------------------------------------------------------------------------------------
dtw_distance = dtw(n,m); 
%dtw_distance = dtw(n,m) / size_warping_path;  % normalized DTW (divided by path length)



