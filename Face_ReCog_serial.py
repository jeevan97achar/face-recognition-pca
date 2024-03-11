# Importing libraries
import io
import numpy as np
import pandas as pd
import matplotlib.image as mat_img
import matplotlib.pyplot as plt
from time import sleep, perf_counter
import warnings
warnings.filterwarnings("ignore")

# Importing images 

# Starting the timer 
start_time = perf_counter()

# Getting file paths
n_img = 10
n_folder = 40
img_path = []
for i in range(1,n_folder+1):
    for j in range(1,n_img+1):
        img_path.append('~/faces/s{}/{}.pgm'.format(i,j))

db_images = []
# Reading the images as 2d arrays
for img in img_path:
    db_images.append(mat_img.imread(img))
height, width = db_images[0].shape

print("Imported images as arrays")

# Flattening the arrays from 2D to 1D
for each in range(len(db_images)):
    db_images[each] = db_images[each].flatten()
    
print("Flattened the arrays from 2D to 1D")

# Creating a pixel*number Dataframes of the images
images_df = pd.DataFrame(data=db_images)
images_df = images_df.T

print("Created a dataframe of flattened arrays")

# Calculating mean of all images
images_df_mean = images_df.mean(axis = 1)

print("Mean of images calculated")

# Mean centering the images
mean_centered_images = pd.DataFrame()
for col in range(len(images_df.columns)):
    mean_centered_images[col] = images_df[col] - images_df_mean

print("Mean centered the image arrays")

# Finding the covariance matrix
image_cov = np.cov(mean_centered_images)
image_cov = np.divide(image_cov, float(len(images_df)))

print("Calculated covariance matrix")

# Finding the eigen values and vectors from the covariance matrix
vals, vecs = np.linalg.eig(image_cov)

print("Calcualted the eigen values and vectors using the covariance matrix")

# Computing the weights of images
image_eigvecs = vecs @ mean_centered_images
image_eigwts = image_eigvecs.T @ mean_centered_images

print("Calculated the eigen weights")

# Calculating eigenfaces of images
image_eigfaces = []
for i in range(n_img*n_folder):
    temp = np.add(np.array(image_eigvecs[i]), np.array(images_df_mean))
    image_eigfaces.append(np.reshape(a=temp,newshape=(height,width)))

print("Calculated the eigen faces")

# Converting the eigen faces to uint8 format
image_eigfaces = np.uint8(image_eigfaces)

print("Converting the eigen faces to uint8 format")

# TEST IMAGE
print("Importing the test image")
test_img = mat_img.imread(fname='~/test_images/6.pgm')
test_img_rs = np.reshape(a=test_img, newshape=(height*width,1))
test_wt = image_eigvecs.T @ (np.subtract(np.float64(test_img_rs.flatten()), images_df_mean))

zero_mat = np.zeros(shape=[n_img*n_folder])

# Calculating the euclidean distance
for i in range(n_img*n_folder):
    sub = np.subtract(test_wt,image_eigwts[i])
    zero_mat[i] = np.linalg.norm(sub)

print("Calculated the euclidean distance between test image and all source images")

# Picking the index of the least distance
zero_mat = zero_mat.tolist()
min_ind = zero_mat.index(min(zero_mat))

print("Found the index of smallest euclidean distance = ",min_ind)
end_time = perf_counter()

print(f' Runtime = {end_time- start_time: 0.2f} seconds')

print("Displaying the test and recognized image")

f = plt.figure()
f.add_subplot(1,2, 1)
plt.title(label="TEST IMAGE")
plt.imshow(test_img, cmap='gray')
f.add_subplot(1,2, 2)
plt.title(label="RECOGNIZED IMAGE")
plt.imshow(np.reshape(a=db_images[min_ind], newshape=(height,width)), cmap='gray')
plt.show()




