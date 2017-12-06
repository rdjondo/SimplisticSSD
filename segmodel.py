from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, UpSampling2D, Activation
from keras.layers import Conv2DTranspose, Add, Flatten, Lambda
from keras.layers import BatchNormalization, Reshape, Permute, Dropout
from keras import backend as K
from keras import regularizers
from keras.models import Model


class SegModel:
    """ Segmentation class  """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.activation_layer = 'selu'
        self.loadVGG16()
        self.buildHead()

    def loadVGG16(self):
        """
        Load VGG 16 as stem model 
        """
        self.base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), 
         pooling=None, include_top=False)
        print('Pre-trained model loaded.')
        for layer in self.base_model.layers:
            layer.trainable = False

        # Get input dimensions
        self.input_height = self.base_model.layers[0].input_shape[1]
        self.input_width = self.base_model.layers[0].input_shape[2]

    def buildHead(self):
        """ Build CNN head of the UNet """
        # Layer 5
        layer_5_1x1 = Conv2D(self.num_classes, 1, padding='same', name='convpool_5_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block5_pool').output)

        up_layer_5 = self.UpLayer(layer_5_1x1)

        # Layer 4
        layer_4_1x1 = Conv2D(self.num_classes, 1, padding='same', name='convpool_4_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block4_pool').output)

        merge_4_and_5 = Add()([up_layer_5, layer_4_1x1])


        up_layer_4_and_5 = Conv2DTranspose(self.num_classes, (3, 3), activation=self.activation_layer,
                                    padding='same', strides=(2,2))(merge_4_and_5)

        up_layer_4_and_5 = BatchNormalization(axis=3, name='up_layer_4_and_5_bn')(up_layer_4_and_5)

        # Layer 3
        layer_3_1x1 = Conv2D(self.num_classes, 1, padding='same', name='convpool_3_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block3_pool').output)

        merge_3_to_5 = Add()([up_layer_4_and_5, layer_3_1x1])

        up_layer_3_to_5 = self.UpLayer(merge_3_to_5)



        # Layer 2
        layer_2_1x1 = Conv2D(self.num_classes, 1, padding='same', name='convpool_2_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block2_pool').output)

        merge_2_to_5 = Add()([up_layer_3_to_5, layer_2_1x1])



        up_layer_2_to_5 = self.UpLayer(merge_2_to_5)


        up_layer_2_to_5 = BatchNormalization(axis=3, 
                                                name='up_layer_2_to_5_bn')(up_layer_2_to_5)

        # Layer 1 UP
        layer_1_1x1 = Conv2D(self.num_classes, 1, padding='same', name='convpool_1_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block1_pool').output)

        merge_1_to_5 = Add()([up_layer_2_to_5, layer_1_1x1])

        up_layer_1_to_5 = self.UpLayer(merge_1_to_5)

        # Final layer
        final_1_1x1 = Conv2D(self.num_classes, 1, padding='same', name='conv_1_1x1',
                    activation=self.activation_layer)(self.base_model.get_layer('block1_conv1').output)

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
        sigmoided_matrix = sigmoid(matrix)
        sum_sig = K.sum(sigmoided_matrix, axis=3)
        sum_sig_reshaped = K.reshape(sum_sig,(-1, self.input_height, self.input_height,1))
        repeat = self.num_classes
        sum_sigmoided_repeated = K.repeat_elements(sum_sig_reshaped, repeat, axis=3)
        softmax_matrix = sigmoided_matrix / sum_sigmoided_repeated
        return softmax_matrix



    # https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/9d0fb65d67334dc332816bcb30d317c2de8b9137/train.py#L23

    def dice_coef(self, y_true, y_pred):
        smooth = 1.0e-3
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(self, y_true, y_pred):
        return -1.0 * self.dice_coef(y_true, y_pred)


    def getModel(self):
        return self.model


    def UpLayer(self, layer):
        simpleUpSample = True
        if not simpleUpSample :
            return Conv2DTranspose(self.num_classes, (3, 3), activation=self.activation_layer,
                                padding='same', strides=(2,2)  )(layer)
        else:
            return UpSampling2D()(layer)


