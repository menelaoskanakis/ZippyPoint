#!/usr/bin/env python3
import tensorflow as tf

from models.modules.modules import Int8Conv2DModule, \
    Int8BinaryShortcutConv2DModule, \
    Int8BinaryShortcutActConv2DModule, \
    Int8PWBinaryShortcutConv2DModule, \
    Int8Conv2DModuleFinal, \
    PixelShuffle, Map2Grid, \
    Interpolate, RescaleToNormalized, BorderMask, \
    BinNormInf
from pathlib import Path


class ZippyPoint(tf.keras.Model):
    def __init__(self, descriptor_size=256, activation="hard_swish", binary_activation="relu", qmax=6.0,
                 use_batch_norm=True, use_dropout=False, bn_momentum=0.9, bin_norm_k=64, **kwargs):
        super().__init__(**kwargs)
        ## Params
        self.descriptor_size = descriptor_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.bn_momentum = bn_momentum

        c_in, c1, c2, c3, c4, c5, d1 = 32, 32, 64, 128, 256, 256, 512
        conv_strides = (1, 1)
        conv_kernels = (3, 3)
        padding = 'same'

        ## Layers
        # Shared Encoder
        self.conv_input = Int8Conv2DModule(c_in, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                           activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum,
                                           qmax=1.0)
        self.conv1a = Int8BinaryShortcutConv2DModule(c1, kernel_size=conv_kernels, strides=conv_strides,
                                                     padding=padding, activation=binary_activation,
                                                     batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv1b = Int8BinaryShortcutActConv2DModule(c1, kernel_size=conv_kernels, strides=conv_strides,
                                                        padding=padding, activation=binary_activation,
                                                        shortcut_activation=activation, batch_norm=use_batch_norm,
                                                        bn_momentum=bn_momentum, qmax=qmax)
        self.conv2a = Int8BinaryShortcutConv2DModule(c2, kernel_size=conv_kernels, strides=conv_strides,
                                                       padding=padding, activation=binary_activation,
                                                       batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv2b = Int8BinaryShortcutActConv2DModule(c2, kernel_size=conv_kernels, strides=conv_strides,
                                                        padding=padding, activation=binary_activation,
                                                        shortcut_activation=activation, batch_norm=use_batch_norm,
                                                        bn_momentum=bn_momentum, qmax=qmax)
        self.conv3a = Int8BinaryShortcutConv2DModule(c3, kernel_size=conv_kernels, strides=conv_strides,
                                                       padding=padding, activation=binary_activation,
                                                       batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv3b = Int8BinaryShortcutActConv2DModule(c3, kernel_size=conv_kernels, strides=conv_strides,
                                                        padding=padding, activation=binary_activation,
                                                        shortcut_activation=activation, batch_norm=use_batch_norm,
                                                        bn_momentum=bn_momentum, qmax=qmax)
        self.conv4a = Int8PWBinaryShortcutConv2DModule(c4, kernel_size=conv_kernels, strides=conv_strides,
                                                       padding=padding, activation=binary_activation,
                                                       batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv4b = Int8BinaryShortcutActConv2DModule(c4, kernel_size=conv_kernels, strides=conv_strides,
                                                        padding=padding, activation=binary_activation,
                                                        shortcut_activation=activation, batch_norm=use_batch_norm,
                                                        bn_momentum=bn_momentum, qmax=qmax)

        # Score head
        self.conv_Sa = Int8Conv2DModule(c5, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                    activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv_Sb = tf.keras.layers.Conv2D(filters=1, kernel_size=conv_kernels, strides=conv_strides,
                                              padding=padding, use_bias=True, activation='sigmoid')

        # Position head
        self.conv_Pa = Int8Conv2DModule(c5, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                    activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv_Pb = tf.keras.layers.Conv2D(filters=2, kernel_size=conv_kernels, strides=conv_strides,
                                              padding=padding, use_bias=True, activation='tanh')

        # Descriptor head
        self.conv_Da = Int8Conv2DModule(c5, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                    activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv_Db = Int8Conv2DModule(d1, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                    activation=None, batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv_Dc = Int8Conv2DModule(c5, kernel_size=conv_kernels, strides=conv_strides, padding=padding,
                                    activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum, qmax=qmax)
        self.conv_Dd = Int8Conv2DModuleFinal(filters=descriptor_size, kernel_size=conv_kernels, strides=conv_strides,
                                              padding=padding, use_bias=True, activation=None, qmax=qmax)
        self.binary_activation = BinNormInf(k=bin_norm_k)

        # Other
        self.pool1 =  Int8Conv2DModule(c1, kernel_size=(1,1), strides=(1, 1), padding=padding,
                         activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum,
                         qmax=1.0)
        self.pool2 = Int8Conv2DModule(c2, kernel_size=(1, 1), strides=(1, 1), padding=padding,
                                      activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum,
                                      qmax=qmax)
        self.pool3 = Int8Conv2DModule(c3, kernel_size=(1, 1), strides=(1, 1), padding=padding,
                                      activation=activation, batch_norm=use_batch_norm, bn_momentum=bn_momentum,
                                      qmax=qmax)
        self.dropout = tf.keras.layers.Dropout(rate=0.2) if self.use_dropout else None
        self.mask_border = BorderMask()
        self.map2grid = Map2Grid(cell_size=8, cross_ratio=2.0)
        self.pos_norm = RescaleToNormalized(cell_size=8)
        self.upsample = PixelShuffle(upscale_factor=2)
        self.desc_interpolate = Interpolate()

    def call(self, x, training=False):

        # Shared
        shared = self.conv_input(x, training)
        shared = tf.nn.space_to_depth(shared, 2)
        shared = self.pool1(shared, training)
        shared = self.conv1a(shared, training)
        shared = self.conv1b(shared, training)
        shared = self.dropout(shared, training) if self.dropout else shared

        shared = tf.nn.space_to_depth(shared, 2)
        shared = self.pool2(shared, training)
        shared = self.conv2a(shared, training)
        shared = self.conv2b(shared, training)
        shared = self.dropout(shared, training) if self.dropout else shared

        shared = tf.nn.space_to_depth(shared, 2)
        shared = self.pool3(shared, training)
        shared = self.conv3a(shared, training)
        skip = self.conv3b(shared, training)
        skip = self.dropout(skip, training) if self.dropout else skip

        shared = self.conv4a(skip, training)
        shared = self.conv4b(shared, training)
        shared = self.dropout(shared, training) if self.dropout else shared

        # Scores
        scores = self.conv_Sa(shared, training)
        scores = self.dropout(scores, training) if self.dropout else scores
        scores = self.conv_Sb(scores)
        scores = self.mask_border(scores)

        # Positions
        positions = self.conv_Pa(shared, training)
        positions = self.dropout(positions, training) if self.dropout else positions
        positions = self.conv_Pb(positions)
        positions = self.map2grid(positions)


        # Descriptors
        descriptors = self.conv_Da(shared, training)
        descriptors = self.dropout(descriptors, training) if self.dropout else descriptors
        descriptors = self.conv_Db(descriptors, training)
        descriptors = self.upsample(descriptors)
        descriptors = self.conv_Dc(descriptors, training)
        descriptors = self.conv_Dd(descriptors, training)

        if not training:
            positions_norm = self.pos_norm(positions)

            # Interpolated descriptors
            descriptors = self.desc_interpolate(descriptors, positions_norm)

            # Normaliza descriptors
            descriptors = self.binary_activation(descriptors)

        return scores, positions, descriptors


def load_ZippyPoint(pretrained_path, model_config={}, input_shape = [240, 320]):
    device_id = tf.config.list_logical_devices('CPU')[0].name

    with tf.device(device_id):
        # Weights dir
        pretrained_path = Path(pretrained_path)
        weights = str(pretrained_path.joinpath("variables/variables").absolute())

        # Define models and perform dummy forward pass
        model = ZippyPoint(**model_config)
        dummy_x = tf.random.uniform((1, )+tuple(input_shape)+(3, ), 0, 1, dtype=tf.float32)
        _ = model(dummy_x, False)

        model.load_weights(weights).expect_partial()

    return model