from tensorflow.keras import Model


class Conv2DBlock(Model):
    def __init__(self,
                 layers_conv2d=None,
                 conv2d_padding='valid',
                 *args,
                 **kwargs):
        """ Constructor

            Args:
                layers_conv2d (list(tuple(str, dict))): configs of conv2d layer. list of tuple(op_name, op_args).
                conv2d_padding (str): padding option of conv2d (valid or same)
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._layers = []

        conv2d_args = {
            "strides": 1,
            "padding": conv2d_padding,
            "activation": 'relu'
        }

        from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

        for layer, args in layers_conv2d:
            if layer == 'conv2d':
                layer_args = args.copy()
                layer_args.update(conv2d_args)
                self._layers.append(Conv2D(**layer_args))
            elif layer == 'maxpooling2d':
                self._layers.append(MaxPooling2D(**args))
            elif layer == 'upsampling2d':
                self._layers.append(UpSampling2D(**args))
            else:
                raise ValueError(f"{layer} is not implemented")

    def call(self, input_tensor, training=False):
        x = input_tensor
        for op in self._layers:
            x = op(x, training=training)
        return x
