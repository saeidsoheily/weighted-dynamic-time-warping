__author__ = 'Saeid SOHILY-KHAH'
%Read the data from a .txt file and Load the dataSet
%-------------------------------------------------------------------------------------------
function [dataSet, k, initialClusters] = function_read_dataset()
format long;
dataSet = load('dataSet.txt');  
initialClusters = dataSet(:,1);
dataSet = dataSet(:,2:end);
k = max(initialClusters);

