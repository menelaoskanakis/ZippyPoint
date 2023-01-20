import tensorflow as tf
import larq

from larq import utils, quantizers

### Binary Normalization Layer ###

@utils.register_alias("bp_inf")
@utils.register_keras_custom_object
class BinNormInf(tf.keras.layers.Layer):
    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        C = input_shape[-1]
        if self.k > C:
            raise ValueError('Descriptor size must be greater than "k". ''Received: %s' % (C, ))

    @tf.function
    def call(self, x):
        B, H, W, C = x.shape
        top_k, top_k_idx = tf.nn.top_k(x, k=self.k, sorted=False)
        top_k_idx = tf.reshape(top_k_idx, (-1, H*W*self.k))
        top_k_idx = tf.squeeze(top_k_idx)

        i = tf.range(0, H)
        j = tf.range(0, W)

        r = tf.repeat(i, W*self.k)
        s = tf.repeat(j, self.k)
        s = tf.tile(s, [H])

        z = tf.zeros((H*W*self.k), dtype=tf.int32)

        l = tf.stack([z,r,s,top_k_idx], axis=-1)
        l = tf.expand_dims(l, axis=0)

        y = tf.scatter_nd(indices=l, updates=tf.ones((1,H*W*self.k)), shape=(B, H, W, C))        # shape of indices: (1,30*40*k,4), updates:(1,30*40*k), shape=(1, 30, 40, 256)

        return y

    def get_config(self):
        config = super().get_config()
        config.update({'k': self.k})
        return config

@utils.register_alias("bp_train")
@utils.register_keras_custom_object
class BinNormTrain(tf.keras.layers.Layer):

    def __init__(self, k: int = 64, eps: float = 1e-4, n_iters: int = 10, branch: int = 60, activation: str = 'sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.eps = eps
        self.n_iters = n_iters
        self.branch = branch
        self.activation = get_activation(activation)
        self.delta = 100. if activation == 'soft_sigmoid' else 7.

    def build(self, input_shape):
        self.ls = tf.linspace(0., 1., self.branch)

    @tf.function
    def call(self, x):
        return self.binnorm_gradient(x)

    @tf.custom_gradient
    def binnorm_gradient(self, x):
        x_sorted = tf.sort(x, axis=-1, direction='DESCENDING')

        nu_lower = -x_sorted[:,self.k-1] - self.delta
        nu_upper = -x_sorted[:,self.k] + self.delta

        def identity(nu_lower, nu_upper):
            return nu_lower, nu_upper

        def rest(nu_lower, nu_upper, r, I):
            nus_a = tf.expand_dims(r, axis=-1) * self.ls
            nus_b = tf.expand_dims(nu_lower, axis=-1)

            nus = nus_a + nus_b

            _xs = tf.expand_dims(x, axis=1) + tf.expand_dims(nus, axis=-1)
            fs = self.activation(_xs)
            fs = tf.reduce_sum(fs, axis=-1) - self.k

            fs = tf.reduce_sum(tf.cast(fs < 0, dtype=tf.int32), axis=-1)
            i_lower = fs - 1
            J = i_lower < 0
            i_lower = tf.where(J, 0, i_lower)
            i_upper = i_lower + 1
            idx_lower = tf.stack([tf.range(tf.shape(i_lower)[0]), i_lower], axis=-1)
            nusg_lower = tf.gather_nd(nus, idx_lower)

            nu_lower = tf.where(I, nusg_lower, nu_lower)                                 # Equivalent implementation

            idx_upper = tf.stack([tf.range(tf.shape(i_upper)[0]), i_upper], axis=-1)
            nusg_upper = tf.gather_nd(nus, idx_upper)
            nu_upper = tf.where(I, nusg_upper, nu_upper)                                 # Equivalent implementation

            nu_lower = tf.where(J, nu_lower-self.delta, nu_lower)

            return nu_lower, nu_upper

        def grad(dy):
            Hinv = 1./(1./y + 1./(1.-y))
            dnu = tf.einsum('bj,bj->b', Hinv, dy)/tf.reduce_sum(Hinv, axis=-1)
            dx = -Hinv*(-dy+tf.expand_dims(dnu, axis=-1))

            return dx

        # Bracketing method
        for i in range(self.n_iters):
            r = nu_upper - nu_lower
            I = r > self.eps
            n_update = tf.reduce_sum(tf.cast(I, tf.int32))

            nu_lower, nu_upper = tf.cond(tf.equal(n_update,0), lambda: identity(nu_lower, nu_upper), lambda: rest(nu_lower, nu_upper, r, I))
        nu = nu_lower + r/2.

        y = self.activation(x + tf.expand_dims(nu, axis=-1))

        return y, grad

    def get_config(self):
        config = super().get_config()
        config.update({'k': self.k, 'eps': self.eps, 'n_iters': self.n_iters, 'branch': self.branch, 'delta':self.delta})

def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

class Interpolate(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, img, grid):
        input_shape = tf.shape(img)
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        # rescale x and y to [0, W-1/H-1]
        max_y = input_shape[1] - 1
        max_x = input_shape[2] - 1
        x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)

        # get pixel value at corner coords
        Ia = get_pixel_value(img, x0, y0)
        Ib = get_pixel_value(img, x0, y1)
        Ic = get_pixel_value(img, x1, y0)
        Id = get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return out

    def get_config(self):
        config = super().get_config()
        return config

class RescaleToNormalized(tf.keras.layers.Layer):

    def __init__(self, cell_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.cell_size = cell_size

    def call(self, positions):
        input_shape = tf.shape(positions)
        global_height = self.cell_size * tf.cast(input_shape[1], dtype=tf.float32)
        global_width = self.cell_size * tf.cast(input_shape[2], dtype=tf.float32)
        pos_x_norm = 2.0 * positions[:,:,:, 0] / (global_width - 1.0) - 1.0
        pos_y_norm = 2.0 * positions[:,:,:, 1] / (global_height - 1.0) - 1.0
        return tf.stack([pos_x_norm, pos_y_norm], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'cell_size': self.cell_size})
        return config

class BorderMask(tf.keras.layers.Layer):

    def call(self, x, img_border=[[0, 0], [1, 1], [1, 1], [0, 0]]):
        input_shape = tf.shape(x)
        pad_size = tf.math.reduce_sum(img_border, axis=-1)
        border_mask = tf.ones((input_shape[0]-pad_size[0],
                               input_shape[1]-pad_size[1],
                               input_shape[2]-pad_size[2],
                               input_shape[3]-pad_size[3]), dtype=tf.float32)
        border_mask = tf.pad(border_mask, img_border)
        return tf.multiply(x, border_mask)

class Map2Grid(tf.keras.layers.Layer):

    # cross_ratio larger 1 allows prediction of keypoint locations across cell borders
    def __init__(self, cell_size: int = 8, cross_ratio: float = 2., **kwargs):       # cell_size==f_downsample, cross_ratio==sigma1
        super().__init__(**kwargs)
        self.cell_size = cell_size
        self.cross_ratio = cross_ratio

    def build(self, input_shapes):
        # multiply constants here so we only have one multiplication in call()
        self.step = tf.constant((self.cell_size-1)/2., dtype=tf.float32)
        self.f = tf.multiply(tf.constant(self.cross_ratio, dtype=tf.float32), self.step)

    def call(self, P_rel):
        input_shape = tf.shape(P_rel)
        W_rel = tf.cast(input_shape[2], dtype=tf.float32)
        H_rel = tf.cast(input_shape[1], dtype=tf.float32)
        W = self.cell_size * W_rel
        H = self.cell_size * H_rel

        c_row = tf.linspace(0., W_rel-1, input_shape[2])
        r_row = tf.linspace(0., H_rel-1, input_shape[1])
        c, r = tf.meshgrid(c_row, r_row)
        cr = tf.stack([c, r], axis=2)
        cr = tf.expand_dims(cr, axis=0)

        P_rel = tf.multiply(P_rel, self.f)
        center_base = tf.multiply(cr, self.cell_size) + self.step
        P_rel = tf.add(center_base, P_rel)

        # Clipping
        pos_x = tf.clip_by_value(P_rel[:,:,:,0], clip_value_min=0., clip_value_max=W-1)
        pos_y = tf.clip_by_value(P_rel[:,:,:,1], clip_value_min=0., clip_value_max=H-1)
        return tf.stack([pos_x, pos_y], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'cell_size': self.cell_size, 'cross_ratio': self.cross_ratio})
        return config

class PixelShuffle(tf.keras.layers.Layer):

    def __init__(self, upscale_factor: int = 2, data_format: str ='NHWC', **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor
        self.data_format = data_format

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.upscale_factor, data_format=self.data_format)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({'upscale_factor': self.upscale_factor, 'data_format': self.data_format})
        return config

### NN Modules ###

class Int8Conv2DModule(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='same', activation=None, batch_norm=True,
                 bn_momentum=0.99, groups=1, qmax=6.0, num_bits=8, **kwargs):
        super().__init__(**kwargs)
        use_bias = True
        if batch_norm:
            use_bias = False
        self.q1 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.q2 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax,
                                                                                                num_bits)) if activation is not None and activation != 'relu' else None
        self.q3 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.conv = larq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                            input_quantizer=None, kernel_quantizer=None, kernel_constraint=None,
                                            use_bias=use_bias, activation=None, groups=groups)
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum) if batch_norm else None

        self.activation = get_activation(activation)

    def call(self, x, training=False):
        x = self.q1(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x, training=training)
        if self.q2:
            x = self.q2(x)
        if self.activation:
            x = self.activation(x)
        x = self.q3(x)
        return x

class Int8Conv2DModuleFinal(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='same', activation=None, use_bias=True,
                 groups=1, qmax=6.0, num_bits=8, **kwargs):
        super().__init__(**kwargs)
        self.q1 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.conv = larq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                            input_quantizer=None, kernel_quantizer=None, kernel_constraint=None,
                                            use_bias=use_bias, activation=None, groups=groups)
        self.q2 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))

        self.activation = get_activation(activation)

    def call(self, x, training=False):
        x = self.q1(x)
        x = self.conv(x)
        x = self.q2(x)
        if self.activation:
            x = self.activation(x)
        return x

class Int8BinaryShortcutConv2DModule(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='same', activation=None, batch_norm=True,
                 bn_momentum=0.99, groups=1, qmax=6.0, num_bits=8, **kwargs):
        super().__init__(**kwargs)
        use_bias = True
        if batch_norm:
            use_bias = False
        self.q1 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.q2 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.conv = larq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                            pad_values=1, input_quantizer="ste_sign", kernel_quantizer="ste_sign",
                                            kernel_constraint="weight_clip", use_bias=use_bias, activation=None,
                                            groups=groups)
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum) if batch_norm else None
        self.add = tf.keras.layers.Add()

        self.activation = get_activation(activation)

    def call(self, x, training=False):
        shortcut = x
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x, training=training)
        x = self.q1(x)
        x = self.add([x, shortcut])
        x = self.q2(x)
        return x

class Int8BinaryShortcutActConv2DModule(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='same', activation=None, shortcut_activation=None,
                 batch_norm=True, bn_momentum=0.99, groups=1, qmax=6.0, num_bits=8, **kwargs):
        super().__init__(**kwargs)
        use_bias = True
        if batch_norm:
            use_bias = False
        self.q1 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.q2 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax,
                                                                                                num_bits)) if shortcut_activation is not None and shortcut_activation != 'relu' else None
        self.q3 = tf.keras.layers.Lambda(
            lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.conv = larq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                            pad_values=1, input_quantizer="ste_sign", kernel_quantizer="ste_sign",
                                            kernel_constraint="weight_clip", use_bias=use_bias, activation=None,
                                            groups=groups)
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum) if batch_norm else None
        self.add = tf.keras.layers.Add()

        self.activation = get_activation(activation)
        self.shortcut_activation = get_activation(shortcut_activation)

    def call(self, x, training=False):
        shortcut = x
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x, training=training)
        x = self.q1(x)
        x = self.add([x, shortcut])
        if self.q2:
            x = self.q2(x)
        if self.shortcut_activation:
            x = self.shortcut_activation(x)
        x = self.q3(x)
        return x

class SimpleNMS(tf.keras.layers.Layer):

    def __init__(self, nms_window=3, **kwargs):
        super().__init__(**kwargs)

        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=nms_window, strides=1, padding='same')

    def call(self, scores, training=False):
        zeros = tf.zeros_like(scores)
        max_mask = scores == self.max_pool(scores)
        for _ in range(2):
            supp_mask = self.max_pool(tf.cast(max_mask, scores.dtype)) > 0
            supp_scores = tf.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == self.max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return tf.where(max_mask, scores, zeros)

class Int8PWBinaryShortcutConv2DModule(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='same', activation=None, shortcut_activation=None, batch_norm=True, bn_momentum=0.99, groups=1, qmax=6.0, num_bits=8, **kwargs):
        super().__init__(**kwargs)
        use_bias = True
        if batch_norm:
            use_bias = False
        self.q1 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.convint8 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=strides, padding=padding, use_bias=True, activation=None)
        self.q2 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.q3 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.q4 = tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(x, -qmax, qmax, num_bits))
        self.conv = larq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, pad_values=1, input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip", use_bias=use_bias, activation=None, groups=groups)
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum) if batch_norm else None
        self.add = tf.keras.layers.Add()

        self.activation = get_activation(activation)
        self.shortcut_activation = get_activation(shortcut_activation)

    def call(self, x, training=False):
        shortcut = x
        shortcut = self.convint8(shortcut)
        shortcut = self.q1(shortcut)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x, training=training)
        x = self.q2(x)
        x = self.add([x, shortcut])
        x = self.q3(x)
        if self.shortcut_activation:
            x = self.shortcut_activation(x)
        x = self.q4(x)
        return x

### Activations ###

@utils.register_alias("ste_heaviside_where")
@utils.register_keras_custom_object
class SteHeavisideWhere(quantizers._BaseQuantizer):
    precision = 1

    def __init__(self, clip_value: float = 1.0, **kwargs):
        self.clip_value = clip_value
        super().__init__(**kwargs)

    def call(self, inputs):
        outputs = ste_heaviside(inputs, clip_value=self.clip_value)
        return super().call(outputs)

    def get_config(self):
        return {**super().get_config(), "clip_value": self.clip_value}

def ste_heaviside(x: tf.Tensor, clip_value: float = 1.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return quantizers._clipped_gradient(x, dy, clip_value)

        return heaviside(x), grad

    return _call(x)

def get_activation(activation="leaky_relu"):
    if activation is None:
        return None
    if isinstance(activation, str):
        if activation == "leaky_relu":
            activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        elif activation == "prelu":
            activation = tf.keras.layers.PReLU()
        elif activation == "relu6":                             # min(max(x, 0), 6)
            activation = tf.nn.relu6
        elif activation == "hard_swish":                        # x*relu6(x+3)/6
            activation = HardSwish()
        elif activation == "soft_sigmoid":                      # 0.5 + 0.5*x/(1+abs(x))
            activation = SoftSigmoid()
        elif activation == "soft_sign":
            activation = SoftSign()
        elif activation == "ste_heaviside":
            activation = lq.quantizers.SteHeaviside(clip_value=1.0)
        elif activation == "ste_heaviside_where":
            activation = SteHeavisideWhere(clip_value=1.0)
        else:
            activation = tf.keras.layers.Activation(activation) # sigmoid, hard_sigmoid, tanh, relu: max(x,0), swish: x*sigmoid(x)
        return activation
    elif isinstance(activation, tf.keras.layers.Layer):
        return activation
    else:
        raise ValueError('activation must either be of type `str` or callable type `tf.keras.layers.Layer`. ''Received: %s' % (type(activation),))

@utils.register_alias("hard_swish")
@utils.register_keras_custom_object
class HardSwish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        self.three = tf.constant(3.)
        self.six = tf.constant(0.166666667)
        self.relu6 = tf.nn.relu6

    def get_config(self):
        config = super().get_config()
        config.update({'trainable': self.trainable})
        return config

    def call(self, x):
        return x * self.relu6(x + self.three) * self.six

@utils.register_alias("soft_sigmoid")
@utils.register_keras_custom_object
class SoftSigmoid(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        self.half = tf.constant(0.5)

    def get_config(self):
        config = super().get_config()
        config.update({'trainable': self.trainable})
        return config

    def call(self, x):
        return self.half + self.half * tf.nn.softsign(x)    # softsign(x): x/(1+abs(x))

@utils.register_alias("soft_sign")
@utils.register_keras_custom_object
class SoftSign(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update({'trainable': self.trainable})
        return config

    def call(self, x):
        return tf.nn.softsign(x)    # softsign(x): x/(1+abs(x))
