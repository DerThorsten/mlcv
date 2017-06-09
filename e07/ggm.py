from __future__ import print_function, division


import skimage.data
import numpy
import scipy.sparse
import operator
import pylab

# some functions
def normImg01(img):
    out = img.copy()
    if img.ndim == 3:
        for c in range(img.shape[2]):
            imgC = out[:, :, c]
            imgC -= imgC.min()
            imgC /= imgC.max()
    elif img.ndim == 2:
        out -= out.min()
        out /= out.min()
    else:
        raise RuntimeError("input has wrong dimension")
    return out

def makeQMatSimple(img):

    imgShape = img.shape[0:2]
    nPixel = imgShape[0] * imgShape[1]
    qMat = scipy.sparse.lil_matrix( (nPixel,nPixel), dtype='float32')

    def vi(x0, x1):
        #return x0*imgShape[1] + x1
        return x1*imgShape[0] + x0


    for x0 in range(imgShape[0]):
        for x1 in range(imgShape[1]):

            vi0 = vi(x0,x1)

            # diagonal value
            qMat[vi0, vi0] = 4.0

            # right neighbor
            if x0 + 1 < imgShape[0]:
                vi1 = vi(x0 + 1,x1)
                qMat[vi0, vi1] = -1.0
                qMat[vi1, vi0] = -1.0

            # lower neighbor
            if x1 + 1 < imgShape[1]:
                vi1 = vi(x0, x1 + 1)
                qMat[vi0, vi1] = -1.0
                qMat[vi1, vi0] = -1.0

    return qMat

def makeQMatFancy(img, gamma):

    def colorToVal(c0, c1):
        #global gamma
        
        diffNorm =  numpy.linalg.norm(c0-c1)
        val = numpy.exp(-gamma*diffNorm)
        #print(c0,c1,diffNorm, val)
        return -1.0*val

    imgShape = img.shape[0:2]
    nPixel = imgShape[0] * imgShape[1]
    qMat = scipy.sparse.lil_matrix( (nPixel,nPixel), dtype='float32')

    def vi(x0, x1):
        return x0*imgShape[1] + x1

    for x0 in range(imgShape[0]):
        for x1 in range(imgShape[1]):

            c0 = img[x0, x1]
            vi0 = vi(x0,x1)

            # right neighbor
            if x0 + 1 < imgShape[0]:
                vi1 = vi(x0 + 1,x1)
                c1 = img[x0+1, x1]
                val = colorToVal(c0, c1)
                qMat[vi0, vi1] = val
                qMat[vi1, vi0] = val

            # lower neighbor
            if x1 + 1 < imgShape[1]:
                vi1 = vi(x0, x1 + 1)
                rgb0 = img[x0, x1+1]
                qMat[vi0, vi1] = val
                qMat[vi1, vi0] = val


    for x0 in range(imgShape[0]):
        for x1 in range(imgShape[1]):
            vi0 = vi(x0,x1)
            s = 0.0
            if x0 + 1 <  imgShape[0]:
                s += qMat[vi0,  vi(x0+1,x1)]
            if x0 - 1 >= 0:
                s += qMat[vi0, vi(x0-1,x1)]
            if x1 + 1 <  imgShape[1]:
                s += qMat[vi0, vi(x0,x1+1)]
            if x1 - 1 >= 0:
                s += qMat[vi0, vi(x0,x1-1)]
            qMat[vi0, vi0] = numpy.abs(s)

    return qMat


def optimize(mu, qMat, sigma):
    
    # mu = x * ( 1 + sigma^2 * qMat)
    aMat = scipy.sparse.identity(qMat.shape[0]) + sigma**2 * qMat

    # tries to solve  Ax=b
    x = scipy.sparse.linalg.spsolve(A=aMat, b=mu)
    return x


# some parameters
noiseLevel = 0.01
gamma = 30.1
sigma = 15.75

# load image
img = skimage.data.astronaut().astype('float32')#[0:100,0:100,:]
img = numpy.swapaxes(img, 0, 1)
# normalize to [0,1]
img = normImg01(img)[0:300,0:300]

# generate noise image
noise = numpy.random.normal(0.0, scale=noiseLevel, size=img.size)

# generate noisy image
noisyImg = numpy.clip(img + noise.reshape(img.shape),0,1)
# or normalize
#noisyImg = numpy.clip(img + noise.reshape(img.shape),0,1)

# generate q matrix
qMat = makeQMatFancy(img, gamma=gamma)
#
# generate q matrix
#qMat = makeQMatSimple(img)#, gamma=gamma)
# optimize


resultImg = noisyImg.copy()
for c in range(3):
    imgC = noisyImg[:,:,c]
    resC = optimize(mu=imgC.ravel(), qMat=qMat, sigma=sigma).reshape(imgC.shape)
    resultImg[:,:,c] = resC

# normalize
resultImg = normImg01(resultImg)

    
if True:

    figure = pylab.figure()

    figure.add_subplot(1,3,1)
    pylab.imshow(img)
    pylab.title("Input Image")

    figure.add_subplot(1,3,2)
    pylab.imshow(noisyImg)
    pylab.title("Noisy Image")

    figure.add_subplot(1,3,3)
    pylab.imshow(resultImg)
    pylab.title("Result Image")

    pylab.show()