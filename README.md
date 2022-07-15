# MNIST-kmeans-pca-gaussian_bayes

Programming exercise in Python which includes the following tasks:
- Load the MNIST dataset and retain only the images corresponding to the digits $i=1,3,7,9$.
- Reshape the images to $28 \times 28$.
- From each image, extract a 2D feature vector **m** where:
  - The first feature component is the mean pixel value of all image matrix rows with an odd index.
  - The first feature component is the mean pixel value of all image matrix columns with an even index.
- Visualize the 2D feature vectors using a scatter plot and assign different colors to each class.
- Apply the Maximin algorithm with $k=4$ on the 2D feature vectors to find 4 cluster centers.
- Using these 4 initial centers, apply the K-Means algorithm on the 2D feature vectors.
- Visualize the 2D feature vectors in a scatter plot and assign different colors to each cluster.
- Calculate the K-Means algorithm purity.
- Apply the PCA algorithm to reduce the dimensions of the $28 \times 28$ training images to $1 \times 1$. 
- Visualize the $1 \times 1$ images using a scatter plot and assign different colors to each class.
- Apply the Maximin algorithm with $k=4$ on the $1 \times 1$ images to find 4 cluster centers.
- Using these 4 initial centers, apply the K-Means algorithm on the $1 \times 1$ images.
- Visualize the $1 \times 1$ images using a scatter plot and assign different colors to each cluster.
- Do the same for $V=25,50,100$, where $V$ is the number of dimensions used for the PCA algorithm (if $V=25$, the reduced-sized images will be of shape $5 \times 5$) without visualizing the reduced-size images.
- Find which value of $V$ maximizes the K-Means clustering purity ($V_{max}$).
- Train a Gaussian Naive Bayes Classifier on the reduced images with size $V_{max}$.
- Apply the PCA algorithm for $V=V_{max} on the test set.
- Evaluate the classifier on the reduced-sized test set and calcualte its classification accuracy score.

