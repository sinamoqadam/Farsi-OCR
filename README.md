# Overview

This is a very simple implementation of Optical Character Recognition. It is written in Python, using Keras Library.

To use the CNN, you need to install Keras Library. To setting it up, follow this link:
<https://github.com/hsekia/learning-keras/wiki/How-to-install-Keras-to-Ubuntu-18.04>

After you've installed Keras, you can use the following command to train, and use the CNN.

    python detect.py
    
Doing so, you'll see something like this:

    59008/60000 [============================>.] - ETA: 0s - loss: 0.1040 - acc: 0.9
    59264/60000 [============================>.] - ETA: 0s - loss: 0.1039 - acc: 0.9
    59520/60000 [============================>.] - ETA: 0s - loss: 0.1039 - acc: 0.9
    59776/60000 [============================>.] - ETA: 0s - loss: 0.1042 - acc: 0.9
    59904/60000 [============================>.] - ETA: 0s - loss: 0.1041 - acc: 0.9
    60000/60000 [==============================] - 17s 282us/step - loss: 0.1042 - acc: 0.9705 - val_loss: 0.1950 - val_acc: 0.9405
    evaluation: 
    Test accuracy 0.94055
    predicting on remaining images ...
    ----------------------------------------

    input[ 0 ].shape = (HEIGHT, WIDTH): (38, 20)

    label[ 0 ]:  2
    prediction:  2
    prediction prob:  0.7294875
    

architecture.cfg contains details about batch size, epochs, optimizer functions, and etc., as well as network structure layers.
To modify CNN structure, you can change the architecture.cfg or make new config files in its format, and pass it to network:

    python detect.py --cfg "path/to/configFile.cfg"
    


You can also load custome dataset. At first this network used to work on MNIST database of handwritten digits: <http://yann.lecun.com/exdb/mnist/>
So it shouldn't be a problem, but if you need help, let me know.
