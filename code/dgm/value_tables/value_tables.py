from __future__ import print_function, division, absolute_import
import numpy
import numbers
import itertools


def conf_gen(shape):
    """
    @brief      yield all configurations
    @param      shape  The shape
    """
    for x in itertools.product(*tuple(range(s) for s in shape)):
        yield x


class ValueTableBase(object):

    def argmin(self):

        min_val = float('inf')
        min_conf = None

        for conf in conf_gen(self.shape):
            val = self[conf]
            if val < min_val:
                min_val = val
                min_conf = conf
        return conf

    def min(self):
        min_val = float('inf')
        for conf in conf_gen(self.shape):
            val = self[conf]
            if val < min_val:
                min_val = val
        return min_val

   

class DenseValueTable(ValueTableBase):
    def __init__(self, values):
        super(DenseValueTable, self).__init__()
        self._values = numpy.require(values,dtype='float32',requirements=['C'])

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    def __getitem__(self, labels):
        if self._values.ndim == 1:
            if isinstance(labels, numbers.Integral):
                return float(self._values[labels])
            else:
                return float(self._values[labels[0]])
        else:
            return float(self._values[tuple(labels)])

    # specializations
    def argmin(self):
        argmin_flat = numpy.argmin(self._values)
        return numpy.unravel_index(argmin_flat, self._values.shape)
    
    def min(self):
        return float(self._values.min())


class PottsFunction(ValueTableBase):
    def __init__(self, shape, beta):
        super(PottsFunction, self).__init__()
        self.beta = beta
        self._shape = shape

    def __getitem__(self, labels):
        assert len(labels) == 2
        return [0.0,self.beta][labels[0]!=labels[1]]

    @property
    def ndim(self):
        return 2
        
    @property
    def shape(self):
        return self._shape



    # specializations  => they are not needed but nice to have
    def argmin(self):
        return (0,0)

    def min(self):
        return 0.0

    
