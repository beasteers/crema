#!/usr/bin/env python
'''Custom Keras layers'''

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


__all__ = ['SqueezeLayer']


class SqueezeLayer(Layer):
    '''
    Keras squeeze layer: remove a length=1 axis
    '''

    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        return (tuple(input_shape[:self.axis])
                + tuple(input_shape[self.axis+1:]))

    def call(self, x, mask=None):
        return K.squeeze(x, axis=self.axis)

    def get_config(self):
        return dict(super(SqueezeLayer, self).get_config(), axis=self.axis)
