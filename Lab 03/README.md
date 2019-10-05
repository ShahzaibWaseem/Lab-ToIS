# Lab 03: Fuzzy logic using scikit-fuzzy

## Task
Your task is to apply the fuzzy version of k-means clustering algorithm. Please follow the following steps:-
1. Firstly generate 400 2D data points using normal Gaussian distribution. Please note that the value of sigma should lie between 0 to 1.5.Also plot the data points for better visualization. Initially you may assume that your data has to be plotted in the form of three clusters.
2. In traditional clustering algorithms the issue at hand is always how to find out number of clusters which is never pre-defined. In this scenario you have to try to cluster your data many times to get a better intuition (may be five or six different times).
	- Hint:-You may use command fuzz.cluster.cmeans to run c-means algorithm with different number of clusters.
3. In this step you have to finalize the number of clusters into which you have to group your data. For this you may use FPC(fuzzy partition coefficient). You have to explore it yourself to understand first and then implement.
4. Once you know the exact number of clusters, your next step is to train C-means using optimal number of clusters which you found in previous steps.
5. Next step is to generate test data of your choice.
6. In last step you have to perform testing using test data.
	- Hint: - you may use cmeans_predict() function. Rest you have to explore on your own.