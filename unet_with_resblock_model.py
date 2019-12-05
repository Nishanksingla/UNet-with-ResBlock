from keras import backend as K
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, UpSampling3D, Concatenate
from keras import initializers


def unet(filter_root, layers, n_class=2, input_size=(10, 256, 256, 1), activation='relu', batch_norm=True, final_activation='softmax'):
    inputs = Input(input_size)
    x = inputs
    # Dictionary for long connections
    var_dict = {}
    # Dictionary for deep supervision
    outputs = []
    # Down sampling
    for i in range(layers):
        out_channel = 2**i * filter_root
        print("out_channel downsampling: {}".format(out_channel))

        # Residual/Skip connection
        res = Conv3D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv3D(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
        if batch_norm:
            conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
        act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv3D(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
        if batch_norm:
            conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

        resconnection = Add(name="Add{}_1".format(i))([res, conv2])

        act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

        # Max pooling
        if i < layers - 1:
            var_dict[str(i)] = act2
            x = MaxPooling3D(padding='same', name="MaxPooling{}_1".format(i))(act2)
        else:
            x = act2
    print("\n")
    # Upsampling
    for i in range(layers - 2, -1, -1):
        print("i upsampling: {}".format(i))

        out_channel = 2**(i) * filter_root
        print("out_channel upsampling: {}".format(out_channel))

        # long connection from down sampling path.
        long_connection = var_dict[str(i)]
        print("long_connection: {}".format(long_connection))

        up1 = UpSampling3D(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv3D(out_channel, 2, activation='relu', padding='same', name="upsamplingConv{}_1".format(i))(up1)
        print("up_conv1: {}".format(up_conv1))
        #  Concatenate.
        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

        #  Convolutions
        up_conv2 = Conv3D(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
        up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

        up_conv2 = Conv3D(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

        # Residual/Skip connection
        res = Conv3D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

        resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

        x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

    # Final convolution
    output = Conv3D(n_class, 1, padding='same',
                    activation=final_activation, name='output')(x)
    outputs.append(output)
    return Model(inputs, outputs=outputs, name='U-ResNet')


def CroppingLayer(inputs):
    ref = inputs[1]
    inputs = inputs[0]
    zeros = K.cast(K.zeros((3,)), dtype='int32')
    pad_begin = K.maximum(
        (K.shape(inputs)[1:-1] - K.shape(ref)[1:-1])//2, zeros)
    print("pad_begin: {}".format(pad_begin))
    pad_end = K.minimum(pad_begin + K.shape(ref)[1:-1], K.shape(inputs)[1:-1])
    print("pad_end: {}\n".format(pad_end))

    output = inputs[:, pad_begin[0]:pad_end[0], pad_begin[1]:pad_end[1],
                    pad_begin[2]:pad_end[2], :]
    print("output: {}\n".format(output))
    return output
