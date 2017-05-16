from __future__ import print_function, division, absolute_import

import numpy
import random 
import nifty
import sys

from dgm.models import *
from dgm.solvers import *
from dgm.value_tables import *


import skimage.transform
import matplotlib.pyplot as plt

numpy.random.seed(42)
random.seed(42)

def rot_gt(gt, a):
    fgt = gt.astype(numpy.float32)
    mi,ma = fgt.min(), fgt.max()
    fgt -= mi 
    fgt /= (ma - mi)

    rot = skimage.transform.rotate(fgt,int(a), order=0)
    rot = numpy.clip(rot, 0.0, 1.0)
    rot *= (ma - mi)
    rot += mi
    return rot

def makeDataset(shape=None, n_images=20, noise=1.0):
    imgs = []
    gts = []
    if shape is None:
        shape = (20, 20)
    for i in range(n_images):

        gt_img = numpy.zeros(shape)
        gt_img[0:shape[0]//2,:] = 1

        #gt_img[shape[0]//4: 3*shape[0]//4, shape[0]//4: 3*shape[0]//4]  = 2

        ra = numpy.random.randint(180)
        #print ra 
        gt_img = rot_gt(gt_img, ra)


        # plt.imshow(gt_img)
        # plt.show()

        img = gt_img + numpy.random.random(shape)*float(noise)

        # plt.imshow(img.squeeze())
        # plt.show()

        imgs.append(img.astype('float32'))
        gts.append(gt_img)

        return imgs, gts



def buildModel(raw_data, gt_image, weights): 

    shape = raw_data.shape
    n_var = shape[0] * shape[1]
    variable_space = numpy.ones(n_var)*2
    gm = DiscreteGraphicalModel(variable_space=variable_space)
    
    def vi(x0,x1):
        return x1 + x0+shape[1]


    # add unaries
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):
            pixel_val = raw_data[x0, x1]
            values = [
                abs(pixel_val),
                abs(1.0 - pixel_val)
            ]
            variables = vi(x0,x1)
            gm.add_factor(variables=variables, value_table=DenseValueTable(values))



if __name__ == "__main__":

    noise = 1.5
    x_train, y_train =  makeDataset(shape=(40,40), n_images=3, noise=noise)
    x_test , y_test  =  makeDataset(shape=(40,40), n_images=3, noise=noise)


    n_weights =  10
    weights = numpy.zeros(n_weights)

    # build the graphical models
    gm_train  = [buildModel(x,y, weights) for x,y in zip(x_train, y_train)]
    gm_test   = [buildModel(x,y, weights) for x,y in zip(x_train, y_train)]





    sys.exit()












    
    # make a grid graphical model
    n_labels = 10
    shape = [100, 100]
    beta = 0.1
    n_var = shape[0] * shape[1]
    variable_space = numpy.ones(n_var)*n_labels

    # construct the graphical model
    gm = DiscreteGraphicalModel(variable_space=variable_space)

    # add unaries
    

    def vi(x0,x1):
        """ @brief  convert coordiantes
            to scalar variable index
        """
        return numpy.ravel_multi_index([x0,x1], shape)



    ##################################################################
    # BUILD THE MODEL
    ##################################################################

    # add unaries
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            values = numpy.random.rand(n_labels)
            variables = vi(x0,x1)
            gm.add_factor(variables=variables, value_table=DenseValueTable(values))

    # add second order
    potts_function = PottsFunction(shape=[n_labels, n_labels], beta=beta)
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            if x0 + 1 < shape[0]:
                variables = [vi(x0,x1),vi(x0+1,x1)]
                gm.add_factor(variables=variables, value_table=potts_function)
            if x1 + 1 < shape[1]:
                variables = [vi(x0,x1),vi(x0, x1+1)]
                gm.add_factor(variables=variables, value_table=potts_function)   



    ##################################################################
    # FIND ARGMIN / OPTIMIZE            
    ##################################################################
    
    if False:
        #run Iterated conditional modes
        params = dict(n_iterations=10000000, temp=0.02)
        icm = GibbsSampler(model=gm, **params)
        visitor = Visitor(visit_nth=10000, name='Gibbs',verbose=False, time_limit=10.0)
        approx_argmin = icm.optimize(visitor=visitor)
        print("Gibbs   ",gm.evaluate(approx_argmin))

        # run GraphCut
        params = dict(tolerance = 0.00001)
        graphcut = GraphCut(model=gm, **params)
        visitor = Visitor(name='GraphCut',verbose=False)
        approx_argmin = graphcut.optimize(visitor=visitor)
        print("GraphCut",gm.evaluate(approx_argmin))


    with nifty.Timer("myicm"):
        # run Iterated conditional modes
        params = dict()
        icm = IteratedConditionalModes(model=gm, **params)
        visitor = Visitor(visit_nth=100, name='Icm',verbose=False)
        approx_argmin = icm.optimize(visitor=visitor)
        print("Icm     ",gm.evaluate(approx_argmin))
