# Air Quality Index from Social Media

Step 1

Image Level Feature from CNN

Deep learning has shown very promising results in computer vision, especially in recognition, detection, and segmentation. It is well known that deep convolutional neural networks are able to capture semantic concept. A simple deep learning feature extraction for one image  is to send the image into a CNN (e.g., AlexNet, VGG Net), and then take the second last layer output. This results in a 4096 dimensional feature vector.

So, the first thing I did is to send all of our 4.2 million images into a CNN, and extract a 4096d CNN features for each image (using RCNN code: https://github.com/rbgirshick/rcnn). Then I use K-means to cluster all these feature vectors into 2014 clusters, using L2 distance with approximate nearest neighbour method. This gives us the following result,

http://pages.cs.wisc.edu/~jiaxu/projects/weibo-text-image/website-kmeans-1024/

As we can see, we are able to group images based on semantic concept. Those selfies images are grouped into same clusters, and outdoor images are clustered together. I later manually labeled each cluster with a cluster concept, and saved it in the following doc,

https://docs.google.com/document/d/1c3p5yjsKFnL1lCMROsriNb7V9--D_i_RvuS6KtR0EM4/edit?usp=sharing

The next thing I did is to find hazy clusters using our AQI labels. Remember, we have only one AQI number for a set of image for one particular city on a particular day. So, I assign images in one set with the reported AQI to approximate the AQI label for each image. Then I compute the average AQI for images in each cluster from Kmeans discussed above, and rank each cluster with this average AQI as the following,

http://pages.cs.wisc.edu/~jiaxu/projects/weibo-text-image/website-1024-clusters-aqi/

Surprisingly, this gives us a very encouraging rank. For example, the first  few clusters are actually figures with AQI info. And another few clusters are also with hazy images.

The Next todoes, is perhaps make use of this CNN feature, and train an indoor/outdoor classifier, so that we can remove those non-relevant images.

Step 2:

Sky Segmentation

From the view of computational photography, sky regions are very important to measure air quality. Then I developed a very simple method to find those sky regions. I first use the MCG code (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to find multi scale super-pixels for an image,

http://pages.cs.wisc.edu/~jiaxu/projects/weibo-text-image/website-superpixel/

As we can see, we already get very good big segments for potential sky regions. Then I threshold the output from MCG to 0.4 and merge those small regions using a local search method from http://www.cse.oulu.fi/~erahtu/ObjSegProp/ObjectSegProposals.html. Next, I pick the biggest segments on the top of an image, and visualize as follows

http://pages.cs.wisc.edu/~jiaxu/projects/weibo-text-image/website-sky-segmentation/

As we can see the sky segmentation is already very good.

The next todos is to perform this step for all the  outdoor images we classified. 

Step 3:

Dark Channel

Dark Channel is very useful measurement for image haziness. I took the code from http://www.mathworks.com/matlabcentral/fileexchange/46147-single-image-haze-removal-using-dark-channel-prior

and visualized the dark channel as follows 

http://pages.cs.wisc.edu/~jiaxu/projects/weibo-text-image/haze-estimation/

One todos is to compute some statistical features (mean, variance) from those non-sky regions (using dark channel and depth) and perhaps build a logistic regressor to learn a hazy predictor.


4. 	Weakly supervised Learning


