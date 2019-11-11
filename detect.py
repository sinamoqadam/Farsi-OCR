from cnn import Network
from dataset import Dataset
from keras.utils import to_categorical
import argparse


def arg_parse():
    """
    Parse arguments to the detect module

    """
    parser = argparse.ArgumentParser(description='Farsi digit Detection Network')

    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/architecture.cfg", type=str)

    return parser.parse_args()


args = arg_parse()

#Set up the neural network
print("Preparing network .....")
network = Network(args.cfgfile)
network.compile()

print("Loading input .....")
dataset = Dataset()
x_train, y_train, x_test, y_test = dataset.loadData(network.net_info.input_shape)

# # Encode the data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Training network .....")
network.fit(x_train, y_train, x_test, y_test)

print("evaluation: ")
network.evaluate(x_test, y_test)

x_predict , y_predict = dataset.predictData(network.net_info.input_shape)
predict_images, predict_labels = dataset.predictImages()
print("predicting on remaining images ...")
prediction = network.predict(x_predict)

network.plotPrediction(predict_images, predict_labels, prediction)
