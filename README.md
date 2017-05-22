## Demo Laplacian Eigenmaps

This file uses a toy dataset of points sampled from a "swiss roll". Such a dataset exemplifies a low-dimensional manifold (2D) embedded in a higher dimensional space (3D). Using an algorithm based on the normalized Graph Laplacian, we can embed the data in a new space.

The image below shows the set of points before and after embedding, where each point before and after the embedding is identified by a unique color, which also indicates the topology of the manifold. Notice that the topology of the manifold is preserved and emphasized in the embedding space -- essentially showing that the embedding "unrolls" the swissroll dataset, endowing it with a new, more suitable, coordinate system. 

The concentration that is occuring near the blue points is explained in my paper <a href="http://epubs.siam.org/doi/10.1137/110839370">A Random Walk on Image Patches</a> 
<img src="./screenshot_swissroll_1.png" width="700">
