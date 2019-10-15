from HodaDataset.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import random


class Dataset:
    def __init__(self):
        super(Dataset, self).__init__()

    def loadData(self, input_shape):
        print('Reading Train 60000.cdb ...')
        train_images, train_labels = read_hoda_dataset(dataset_path='./HodaDataset/DigitDB/Train 60000.cdb',
                                                        images_height=input_shape[0],
                                                        images_width=input_shape[1],
                                                        one_hot=False,
                                                        reshape=False)

        print('Reading Test 20000.cdb ...')
        test_images, test_labels = read_hoda_dataset(dataset_path='./HodaDataset/DigitDB/Test 20000.cdb',
                                                        images_height=input_shape[0],
                                                        images_width=input_shape[1],
                                                        one_hot=False,
                                                        reshape=False)

        train_images = train_images.reshape(60000, 28, 28, 1).astype('float32')
        test_images = test_images.reshape(20000, 28, 28, 1).astype('float32')
        train_images /= 255
        test_images /= 255

        return train_images, train_labels, test_images, test_labels

    def predictData(self, input_shape):
        remaining_images, remaining_labels = read_hoda_dataset(dataset_path='./HodaDataset/DigitDB/RemainingSamples.cdb',
                                                        images_height=input_shape[0],
                                                        images_width=input_shape[1],
                                                        one_hot=False,
                                                        reshape=False)
        # remaining_images, remaining_labels = read_hoda_cdb('./HodaDataset/DigitDB/RemainingSamples.cdb')
        remaining_images = remaining_images.reshape(22352, 28, 28, 1).astype('float32')
        remaining_images /= 255

        return remaining_images[:4], remaining_labels[:4]

    def predictImages(self):
        return read_hoda_cdb('./HodaDataset/DigitDB/RemainingSamples.cdb')
