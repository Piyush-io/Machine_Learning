In unsupervised Learning, ***we do not provide the labels***(expected result). The system tries to learn without help.
Most important Unsupervised Learning Algorithms are:
- [[Clustering]]:
	- [[K-Means]]
	- [[DBSCAN]]
	- [[Hierarchical Cluster Analysis]]
- [[Anomaly Detection and Novelty Detection]]:
	- [[One-Class SVM]]
	- [[Isolation Forest]]
- [[Visualisation and Dimensionality Reduction]]:
	- [[Principal Component Analysis]]
	- [[Kernel PCA]]
	- [[Locally Linear Embedding]]
	- [[t-Distributed Stochastic Neighbour Embedding]]
- [[Association rule Learning]]
	- [[Apriori]]
	- [[Eclat]]

***Clustering*** algorithms try to detect groups of similar attributes. If you use hierarchical Clustering Algorithm, you may get even small groups(more sub division).

***Visualisation Algorithms*** are those models where we feed it a lot of data and try to get 2D or 3D representation of the data as output, to see how the data is organised and identify unsuspected patterns that may be overlooked by the system.

***Dimensionality Reduction*** is a task where we try to simply the data without losing too much information. (It is a good practice as it may even improve our Models performance, but it certainly reduces the space required to store the data and the time required to train the model)

 - When we merge related attributes into one single attribute is called ***feature extraction***.

Another task is ***Anomaly Detection***, where we try to find outliers or you can unusual data. Similarly we have ***Novelty Detection***, with only difference that in Novelty detection the model is trained only on normal data while Anomaly Detection may even work with a small percentage of outliers.

***Association Rule Learning*** is also one of the application of Unsupervised Learning which is used when we try to find relation between attributes. 



