# Dimensionality Reduction

Owner: Piyush Mittal

When training Machine Learning models in real, we do not encounter data with only tens, hundreds or even thousands of features, but with millions of features. 

This increased dimensionality not only makes it difficult to visualise (as for most humans even a 4 dimensional hypercube is difficult to comprehend) but it also slows down the training time a lot.

Reducing the Dimensions might help in :

- Data Visualisation(If can reduced to 2 or 3 dimensions)
- Noise Reduction which may help in better performance
- Faster training time

> ***GENERALLY THIS LEADS TO SLIGHTLY WORSE PERFORMANCE***
> 

## The Curse of Dimensionality

Couple of Facts :

1. In smaller dimensions, the probability for a random point to be near a border is very less, but on increasing the dimensions, this probability rises exponentially.
2. As soon as we increase the dimensions, the distance between two randomly picked points increases exponentially.

Since any random points are likely to be very far away, this implies that high-dimensional datasets are likely to have instances that are very far from each other, meaning any new testing instance also has a chance to be far away, making predictions very less reliable.

## Approaches For Dimensionality Reduction

- ***Projection*** - In most real world problems, many features are almost constant and others are highly correlated, meaning they lie near(very close) to each other within a lower dimensional subspace.
    - For example we try to perpendicularly project 3D data onto a 2D subspace (plane), giving us new features, but retaining its correlation.
    - But it is not always the best way to reduce dimensions, as in case of Swiss Roll data where if you simply project data onto a plane will lead to overlapping.
- ***Manifold* -** A Swiss roll is a 2D manifold. It can be twisted and bent into a higher dimensional space. A d dimensional manifold is a part of a n dimensional space where d < n.
    - Assumptions :
        1. Manifold Hypothesis/Assumption , which holds that most real world high dimensional datasets lie close to a much lower dimensional manifold.
        2. The task to be performed will be much simpler if expressed in lower dimensional space of the manifold.
    - The second assumption isn’t necessarily always true as sometimes a higher dimensional data may be easily separable.
    - So, its better to first train your model without dimension reduction as it may not always lead to a better model.

### PCA - Principal Component Analysis

- Most Popular dimensionality reduction algorithm.
- We try to find the closest lying hyperplane near the data and then project the data onto it.
1. ***Preserving the Variance:*** 
    - We try to select the axes that preserves the maximum amount of variance, because doing so we lose less information.
    - The aim of the axes is also to minimise the mean squared distance between the original dataset and their projection on the axis.
2. ***Principal Components:***
    - We start by subtracting the mean of the data from all the instances so as to centre the data and obtain a mean of 0. (When using scikit-learn, it does this for us)
    - Doing the above step, we can start by random line passing through the origin, and adjust it so that it fits the data best(reducing the mean squared distance between the instances and the projection).
    - Once we find the best line, we find the second axis orthogonal to the first one, which will account for the largest amount of the remaining variance.
    - With increase in dimensions, the required PCs also increase, for eg. 2 PCs for a 2D hyperplane.
    - The ***unit vector*** that defines the axis is called the ***principal component***.
    - If we add some noise in the dataset and run PCA again, the PCs obtained would lie on the same axis but may be reversed now. (meaning the plane they define remains the same)
    - To obtain the PCs we use Singular Value Decomposition(SVD), which breaks down our training set matrix X into matrix multiplication of 3 matrices,
        
        $$
                                               X = U*\sum*V^T 
        $$
        
        where $V$ contains the PCs(c1,c2,…,cn).
        
3. ***Projecting Down the Data:***
    - To project down the data onto the hyperplane simply compute the matrix multiplication of the training set $X$ with the matrix $W_d$ which contains the first d (same as the dimensions of our hyperplane) PCs.
4. ***Explained Variance Ratio***:
    - This ratio gives the ratio or amount of variance explained by our Principal Components, or we can say, the amount of variance that lies along each Principal Component.
5. ***Choosing the Right Number of Dimensions***:
    - For Data Visualisation we would generally want to reduce the number of dimensions to 2 or 3 .
    - We can use numpys’ cumsum function to get the element where the cumulative sum of the variance is greater than or equal to 95%, thus later we can set d equal to the argmax of it.
    - Another option is to plot the cumsum, which usually contains an elbow where the explained variance stops growing fast, so around this elbow if reduce the number of dimensions we won’t lose that much information.
    - We can also set the n_components of PCA to values between 0.0 to 1.0. For eg. if n_components = 0.95, we want the number of components that preserve 95% of the variance.
6. ***PCA for Compression*** :
    - We can achieve reasonable compression using PCA. If we take MNIST dataset for example and apply PCA with n_components = 0.95, we reduce the number of features from 784 to just over 150, meaning almost 20% of the original size.
    - This can help speed up classification algorithms like SVM classifier by a lot.
    - We can also decompress the dataset back to 784 dimensions by applying inverse_transform.
        
        > **Doing this won’t give us the original dataset back because we lost around 5% of information during compression.**
        > 
    - The difference between the original dataset and the reconstructed dataset can be calculated using the reconstruction error.

### Randomised PCA

- To speed up the process, we can set svd_solver hyperparameter to “randomized”.
- What is does is that it finds an approximation of the principal components.
- This is used when d is much smaller than n(n > 500 and d is less than 80% of that)
- To use the full SVD, set the mentioned hyperparameter to “full”

### Incremental PCA

- Traditional PCA algorithms require the entire training set to fit in memory, which can be an issue for large datasets. Incremental PCA (IPCA) resolves this issue by loading parts of the data into memory.
- IPCA applies PCA to mini-batches of the dataset, making it suitable for online (on-the-fly) PCA application.
- However, it does require more computational resources due to frequent updates, compared to traditional PCA methods.
- We can use partial fit of inc_pca by using array split of numpy.
- Alternatively we can use memmap, which allows you to manipulate a large array stored in a binary file on a disk as if it were entirely stored in memory.
    - It only load the data on the memory that it needs.
    - Since ICPA uses only small part of an array at any time so the memory usage remains under control.

### Kernel PCA

- As we saw in SVMS, a liner decision boundary in the high dimensional feature space corresponds to a complex non linear decision boundary in the original space.
- Fortunately we can use the kernel trick to perform complex non linear projections for dimensionality reduction.
- It’s often good for preserving the clusters of instances or even unrolling the datasets that lie close to a twisted manifold.
    - ***Selecting A kernel and Turing Hyper parameters :***
        - To find the best kernel and hyper parameters, we can create a pipeline that performs the dimensionality reduction and trains a classification model.
        - We the perform grid search on it, to find which kernel and hyper parameters would lead to the best performance.
        - In Kernel PCA there is no inverse transform method by default, so we need to set fit_inverse_tranform method hyperparameter to “true”.

### Local Linear Embedding

- It is a manifold learning technique that does not rely on projections like the previous algorithms.
- Instead, it first measures how each training instance linearly relates to its closest neighbours, and then looks for a low-dimensional representation of the training set where these local relationships are best preserved.
- This makes LLE particularly good at unrolling twisted manifolds, especially when there is not too much noise.
- ***Working of LLE***:
    - For each instance $x^i$ , the algorithms finds its k nearest neighbours and then tries to reconstruct $x^i$  as linear function of those k neighbours.
    - In more detail, it tries to find the weights $w_i,_j$ such that the square distance betwen $x^(i)$ and $\displaystyle\sum_{j=1}^m w_i,_j*x^j$ is as small as possible, assuming $w_i,_j$  = 0 for $x^j$ is not one of the k closest neighbours of $x^i$.
    
    > $\widehat{W} = argmin_w \displaystyle\sum_{i=1}^m\bigg( x^i - \displaystyle\sum_{i=1}^m(w_i,_j*x^j)\bigg)$
    > 
    - After, this the weight matrix containing the weights encodes the linear relationships between the training instances.
    - The second step is to map the training instances into a d-dimensional space while preserving these local relationships as much as possible, meaning the squared difference between the image $z^i$ and the summation of the weights $\displaystyle\sum_{j=1}^m\hat{w}_i,_jz^j$ to be as small as possible.
    - The above step provides us with another unconstrained optimisation problem but instead of finding the optimal weights for fixed instances, here we fix the weights and try to find optimal position for the instances’ image.
    
    > $\widehat{Z} = argmin_z \displaystyle\sum_{i=1}^m\bigg( z^i - \displaystyle\sum_{j=1}^m(\hat{w}_i,_jz^j)\bigg)$
    > 
    

### Other Dimensionality Reduction Techniques

1. ***Multidimensional Scaling(MDS)*** reduces the dimensionality while trying to preserve the distances between the instances.
2. ***Isomap*** creates a graph connecting each instance to its nearest neighbours, then reduces the dimensionality while trying to preserve the geodesic distances(number of nodes on the shortest path between two nodes) between the instances.
3. ***t-Distributed Stochastic Neighbour Embedding (t-SNE)*** reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. Usually it is used for visualisation.
4. ***Linear Discriminant Analysis (LDA)*** learns the most discriminative axes between the classes. These axes can be used to define a hyperplane onto which to project the data. The project will keep the classes as far apart as possible. Good algorithm to reduce dimensionality before running a classification model like SVM.

# Thank you….