from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, UpSampling2D, Activation
from keras.layers import Conv2DTranspose, Add, Flatten, Lambda
from keras.layers import BatchNormalization, Reshape, Permute, Dropout
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.generic_utils import get_custom_objects

import numpy as np

def activation_layer():
    return LeakyReLU(alpha=1e-2)

get_custom_objects().update({'activation_layer': Activation(activation_layer)})



class SegModel:
    """ Segmentation class  """
    def __init__(self, num_classes, class_weight=None):
        if class_weight is not None:
            self.class_weight = class_weight
        else:
            self.class_weight = np.ones(num_classes)
        self.num_classes = num_classes
        self.loadVGG16()
        self.buildHead()
    
    def loadVGG16(self):
        """
        Load VGG 16 as stem model 
        """
        self.base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), 
         pooling=None, include_top=False)
        print('Pre-trained model loaded.')
        #for layer in self.base_model.layers:
        #    layer.trainable = False

        # Get input dimensions
        self.input_height = self.base_model.layers[0].input_shape[1]
        self.input_width = self.base_model.layers[0].input_shape[2]


    def UpLayer(self, layer):
        simpleUpSample = True
        if not simpleUpSample :
            return Conv2DTranspose(self.num_classes, (3, 3), activation=self.activation_layer,
                                padding='same', strides=(2,2)  )(layer)
        else:
            return UpSampling2D()(layer)
        
    def gen1x1Conv(self, layer_name):
        """layer_name: str of the layer to extract from VGG"""
        layer_i_1x1 = Conv2D(self.num_classes, 1, padding='same', 
                             name='{}_1x1'.format(layer_name)) (self.base_model.get_layer(layer_name).output)
        return activation_layer() (layer_i_1x1)

    def gen1x1ConvUp(self, layer_name):
        """layer_name: str of the layer to extract from VGG"""
        conv_layer_i = self.gen1x1Conv(layer_name)
        return self.UpLayer(conv_layer_i)

    def gen1x1ConvUpMerge(self, layer_name, layer_to_merge):
        """layer_name: str of the layer to extract from VGG
        layer_to_merge: Keras layer"""
        conv_layer_i = self.gen1x1Conv(layer_name)
        merged = Add()([layer_to_merge, conv_layer_i])
        up_layer = self.UpLayer(merged)
        return activation_layer()(up_layer)
        

    def buildHead(self):
        """ Build CNN head of the UNet """
        # Layer 5
        up_layer_5 = self.gen1x1ConvUp('block5_pool')

        # Layer 4
        up_layer_4_and_5 = self.gen1x1ConvUpMerge('block4_pool', up_layer_5)
        up_layer_4_and_5 = BatchNormalization(axis=3, name='up_layer_4_and_5_bn')(up_layer_4_and_5)

        # Layer 3
        up_layer_3_to_5 = self.gen1x1ConvUpMerge('block3_pool', up_layer_4_and_5)

        # Layer 2
        up_layer_2_to_5 = self.gen1x1ConvUpMerge('block2_pool', up_layer_3_to_5)

        up_layer_2_to_5 = BatchNormalization(axis=3, 
                                                name='up_layer_2_to_5_bn')(up_layer_2_to_5)

        # Layer 1 UP
        up_layer_1_to_5 = self.gen1x1ConvUpMerge('block1_pool', up_layer_2_to_5)

        # Final layer
        final_1_1x1 = Conv2D(self.num_classes, 1, padding='same',
                             name='conv_1_1x1')(self.base_model.get_layer('block1_conv1').output)
        
        final_1_1x1 =  activation_layer()(final_1_1x1)
        
        final_merge = Add(name='final_merge')([final_1_1x1, up_layer_1_to_5])

        soft_out = Lambda(self.depth_softmax, name='soft_out')(final_merge)

        self.model = Model(inputs=self.base_model.input, outputs=[final_merge,soft_out])

        self.model.summary()

        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')
        return self.model


    def depth_softmax(self, matrix):
        """
        The softmax activation function doesnt seeem to perform as well as the dice coeff.
        See justification in https://arxiv.org/pdf/1707.04912.pdf
        However, we keep using this layer for displaying the total accuracy of the
        classifier during training and for displaying pixel-wise confidence during
        inference.
        """
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        #sigmoided_matrix_in = sigmoid(matrix)
        sigmoided_matrix_in = K.softplus(matrix)
        weights_np = np.tile(self.class_weight, (self.input_height, self.input_width, 1) ) 
        weights = K.constant(weights_np, dtype='float32')
        print('weights.shape', weights.shape)
        print('sigmoided_matrix_in.shape', sigmoided_matrix_in.shape)
        sigmoided_matrix = sigmoided_matrix_in * weights
        print('sigmoided_matrix.shape', sigmoided_matrix.shape)
        sum_sig = K.sum(sigmoided_matrix, axis=3)
        print('sum_sig.shape', sum_sig.shape)
        sum_sig_reshaped = K.reshape(sum_sig,(-1, self.input_height, self.input_width,1))
        repeat = self.num_classes
        sum_sigmoided_repeated = K.repeat_elements(sum_sig_reshaped, repeat, axis=3)
        softmax_matrix = sigmoided_matrix / sum_sigmoided_repeated
        return softmax_matrix


    def getModel(self):
        return self.model


