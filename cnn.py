# Convolutional Neural Network with Keras

# Import the libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt



def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class NetInfo:
    def __init__(self):
        super(NetInfo, self).__init__()


def create_model(blocks):
    # Captures the information about the input and pre-processing
    net_info = NetInfo()
    net_info.batch_size = int(blocks[0]["batch"])
    net_info.epochs = int(blocks[0]["epochs"])
    net_info.optimizer = blocks[0]["optimizer"]
    net_info.loss = blocks[0]["loss_function"]
    net_info.metrics = blocks[0]["metrics"]
    net_info.verbosity = int(blocks[0]["verbosity"])

    width = int(blocks[0]["width"])
    height = int(blocks[0]["height"])
    channels = int(blocks[0]["channels"])
    input_shape = (height, width, channels)
    net_info.input_shape = input_shape

    model = Sequential()
    firstLayer = True

    for index, x in enumerate(blocks[1:]):
        # If it's a convolutional layer
        if x["type"] == "convolutional":
            # Get the info about the layer
            activation = x["activation"]
            bias = x["use_bias"]
            filters = int(x["filters"])
            pad = x["pad"]
            kernel_size = int(x["size"])
            x["strides"] = x["strides"].split(',')
            height_stride = int(x["strides"][0])
            # width stride, if there exists one.
            try:
                width_stride = int(x["strides"][1])
            except:
                width_stride = height_stride

            strides = (height_stride, width_stride)

            if firstLayer:
                model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad, use_bias=bias,
                                 input_shape=input_shape, activation=activation))
                firstLayer = False
            else:
                model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad, use_bias=bias,
                                 activation=activation))

        # If it's a max pooling layer
        elif x["type"] == "max_pooling":
            pool_size = int(x["pool_size"])
            model.add(MaxPooling2D(pool_size=pool_size))

        # If it is a flatten layer
        elif x["type"] == "flatten":
            model.add(Flatten())

        # shortcut corresponds to skip connection
        elif x["type"] == "dense":
            outputUnits = int(x["output_units"])
            activation = x["activation"]
            model.add(Dense(units=outputUnits, activation=activation))

    return model, net_info


class Network:
    def __init__(self, cfgfile):
        super(Network, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.model, self.net_info = create_model(self.blocks)

    # Compile the model
    def compile(self):
        self.model.compile(optimizer=self.net_info.optimizer, loss=self.net_info.loss, metrics=[self.net_info.metrics])

    # Fit the model to the training data
    def fit(self, training_data, training_label, test_data, test_label):
        self.model.fit(training_data, training_label,
              batch_size = self.net_info.batch_size,
              epochs = self.net_info.epochs,
              verbose = self.net_info.verbosity,
              validation_data = (test_data, test_label))

    # Evaluate the accuracy of the model
    def evaluate(self, test_data, test_label):
        score = self.model.evaluate(test_data, test_label, verbose=0)
        print('Test accuracy', score[1])

    # Predict on input
    def predict(self, input):
        return self.model.predict(input)

    def plotPrediction(self, images, labels, predictionList):
        fig = plt.figure(figsize=(15, 4))
        for i in range(4):
            prediction = predictionList[i]

            maxProb = 0
            maxLabel = 0
            for predictionLabel in range(len(prediction)):
                if maxProb < prediction[predictionLabel]:
                    maxProb = prediction[predictionLabel]
                    maxLabel = predictionLabel

            print('----------------------------------------')
            print()

            print('input[', i, '].shape = (HEIGHT, WIDTH):', images[i].shape)
            print()

            print('label[', i, ']: ', labels[i])
            print('prediction: ', maxLabel)
            print('prediction prob: ', maxProb)
            print()

            fig.add_subplot(1, 4, i + 1)
            plt.title('prediction[' + str(i) + '] = ' + str(maxLabel))
            plt.imshow(images[i], cmap='gray')

        plt.show()


