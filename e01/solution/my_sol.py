from __future__ import print_function

import os
import glob
import numpy
import skimage
import skimage.io

from skimage.transform import resize
from skimage.morphology import disk
from skimage import filters
from matplotlib import pyplot as plt 
from scipy import ndimage, misc, stats
from sklearn.ensemble import RandomForestClassifier
from skimage.morphology import disk
from multiprocessing import cpu_count



def compute_features(img):

    features_imgs = []

    # per channel
    for c in range(3):

        img_c  = img[:, :, c]

        # gaussian
        sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]
        for sigma in sigmas:
            feat = filters.gaussian(img_c, sigma=sigma)
            features_imgs.append(feat[:,:,None])

        # median & gradient
        disk_sizes = [3, 5, 7 , 11]
        for disk_size in disk_sizes:
            feat = filters.rank.median(img_c, disk(disk_size))
            features_imgs.append(feat[:,:,None])

            feat = filters.rank.gradient(img_c, disk(disk_size))
            features_imgs.append(feat[:,:,None])


        # laplace
        ksizes = [3,5,7,11]
        for ksize in ksizes:
            feat = filters.laplace(img_c, ksize=ksize)
            features_imgs.append(feat[:,:,None])

        # sobel
        features_imgs.append(filters.sobel(img_c)[:,:,None])


    # coordinates
    shape = img.shape[0:2]
    x0 = numpy.linspace(0,1,  shape[0])
    x1 = numpy.linspace(0,1, shape[1])
    x0, x1 = numpy.meshgrid(x0, x1)
    x0 = x0.T
    x1 = x1.T
    features_imgs.append(x0[:,:,None])
    features_imgs.append(x1[:,:,None])

    # center diff
    center_dist = ((x0-0.5)**2 + (x1-0.5)**2)**0.5
    features_imgs.append(center_dist[:,:,None])


    return numpy.concatenate(features_imgs, axis=2)



if __name__ == "__main__":


    train_paths = glob.glob("training_set/*")
    test_paths = glob.glob("test_set/*")
    ground_truth_path = glob.glob("ground_truth_train/*")
    ground_truth_test_path = glob.glob("ground_truth_test/*")



    # open the images
    train_raw = [ndimage.imread(f) for f in train_paths]
    test_raw  = [ndimage.imread(f) for f in test_paths]

    train_gt = [ndimage.imread(f) for f in ground_truth_path]
    test_gt  = [ndimage.imread(f) for f in ground_truth_test_path]



    # make the training images have the correct shape
    for i in range(len(train_gt)):
        gt = resize(train_gt[i], train_raw[i].shape[0:2])
        gt = numpy.round(gt,0)
        train_gt[i] = gt.astype('uint8')

    for i in range(len(test_gt)):
        gt = resize(test_gt[i], test_raw[i].shape[0:2])
        gt = numpy.round(gt,0)
        test_gt[i] = gt.astype('uint8')


    train_features = [compute_features(img) for img in train_raw]
    test_features  = [compute_features(img) for img in test_raw]

    n_features = train_features[0].shape[2]

    # flatten features
    

    train_features = [f.reshape([-1,n_features ]) for f in train_features]
    test_features  = [f.reshape([-1,n_features ]) for f in test_features]   


    train_features = numpy.concatenate(train_features, axis=0)


    # flatten labels
    train_gt = [f.reshape([-1]) for f in train_gt]
    train_gt = numpy.concatenate(train_gt, axis=0)

    # train a random forest
    print(train_features.shape, train_gt.shape)




    n_trees = 100
    rf = RandomForestClassifier(n_estimators = n_trees, n_jobs = cpu_count())
    rf.fit(X=train_features, y=train_gt)


    # predict
    for i in range(len(test_raw)):
        
        shape = test_raw[i].shape[0:2] + (2,)
        #print("test_features[i]",test_features[i].shape)
        pred = rf.predict_proba(test_features[i]).reshape(shape)
        print(pred.shape)

        skimage.io.imsave("out%d.png"%i, pred[:,:,0])
        skimage.io.imsave("raw%d.png"%i, test_raw[i])