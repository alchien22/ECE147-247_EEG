import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from braindecode.augmentation import TimeReverse, SmoothTimeMask, SignFlip, FrequencyShift, FTSurrogate, ChannelsDropout, BandstopFilter


def train_data_prep(X, y, sub_sample, average, noise):
    """Preprocesses EEG training data.
    
    Args:
        X: A np.array with EEG training input data.
        y: A np.array with EEG training label data.
        sub_sample: An integer for the subsampling rate.
        average: An integer for averaging size.
        noise: A boolean for adding noise.

    Returns:
        A np.array with preprocessed EEG training data.
    """
    total_X = None
    total_y = None
    
    # Trimming the data (sample, 22, 1000) -> (sample, 22, 500)
    X = X[:, :, 0:800]
    #print('Shape of X after trimming:', X.shape)
    
    # Maxpooling the data (sample, 22, 500) -> (sample, 22, 500 / sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y
    #print('Shape of X after maxpooling:', total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    #print('Shape of X after averaging+noise and concatenating:', total_X.shape)
    
    # Subsampling
    for i in range(sub_sample):
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :, i::sub_sample].shape) if noise else 0.0)
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
    
    
    probability = 0.5 
    
    # Time Reverse
    X_tensor = torch.from_numpy(X[:, :, 0:400])   
    time_reverse_augmentation = TimeReverse(probability)
    X_augmented, y = time_reverse_augmentation.operation(X_tensor, y)
    total_X = np.vstack((total_X, X_augmented))
    total_y = np.hstack((total_y, y))
    
    # Smooth Time Mask
    #X_tensor = torch.from_numpy(X[:, :, 0:400])   
    #smooth_time_mask = SmoothTimeMask(probability)
    #X_augmented, y = smooth_time_mask.operation(X_tensor, y, mask_start_per_sample= mask_len_samples=100)
    #total_X = np.vstack((total_X, X_augmented))
    #total_y = np.hstack((total_y, y)) 
    
    # Sign Flip
    X_tensor = torch.from_numpy(X[:, :, 0:400])   
    sign_flip = SignFlip(probability)
    X_augmented, y = sign_flip.operation(X_tensor, y)
    total_X = np.vstack((total_X, X_augmented))
    total_y = np.hstack((total_y, y)) 
    
    # Frequency Shift
    X_tensor = torch.from_numpy(X[:, :, 0:400])   
    sign_flip = FrequencyShift(probability, sfreq=250)
    X_augmented, y = FrequencyShift.operation(X_tensor, y, delta_freq=2, sfreq=250)
    total_X = np.vstack((total_X, X_augmented))
    total_y = np.hstack((total_y, y)) 
    
    # Fourier transform surrogate
    X_tensor = torch.from_numpy(X[:, :, 0:400])   
    sign_flip = FTSurrogate(probability)
    #channel_indep should be false bc of the eeg data or something they said in the documentation
    X_augmented, y = FTSurrogate.operation(X_tensor, y, phase_noise_magnitude=0.5, channel_indep=False)
    total_X = np.vstack((total_X, X_augmented))
    total_y = np.hstack((total_y, y))
    
    
    # Channels Dropout
    X_tensor = torch.from_numpy(X[:, :, 0:400])   
    sign_flip = ChannelsDropout(probability)
    X_augmented, y = ChannelsDropout.operation(X_tensor, y, p_drop=0.2)
    total_X = np.vstack((total_X, X_augmented))
    total_y = np.hstack((total_y, y)) 
    
    # Bandstop Filter
    # X_tensor = torch.from_numpy(X[:, :, 0:400])   
    # sign_flip = BandstopFilter(probability, sfreq=250)
    # X_augmented, y = BandstopFilter.operation(X_tensor, y, sfreq=250, bandwidth=2, freqs_to_notch)
    # total_X = np.vstack((total_X, X_augmented))
    # total_y = np.hstack((total_y, y)) 
    

    #print('Shape of X after data augmentation:', total_X.shape)
    #print('Shape of Y:', total_y.shape)
    return total_X, total_y

def valid_data_prep(X, y):
    """Preprocesses EEG validation data.
    
    Args:
        X: A np.array with EEG validation input data.
    
    Returns:
        A np.array with preprocessed EEG testing data.
    """
    total_X = None
    total_y = None
    
    # Trimming the data (sample, 22, 1000) -> (sample, 22, 500)
    X = X[:, :, 0:800]
    #print('Shape of X after trimming:', X.shape)
    
    # Maxpooling the data (sample, 22, 500) -> (sample, 22, 500 / sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)

    total_X = X_max
    total_y = y 

    return total_X, total_y

def test_data_prep(X):
    """Preprocesses EEG testing data.
    
    Args:
        X: A np.array with EEG testing input data.
    
    Returns:
        A np.array with preprocessed EEG testing data.
    """
    total_X = None
    
    # Trimming the data (sample, 22, 1000) -> (sample, 22, 500)
    X = X[:, :, 0:800]
    #print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample, 22, 500) -> (sample, 22, 500 / sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    total_X = X_max
    #print('Shape of X after maxpooling:',total_X.shape)
    
    return total_X


def to_categorical(y, num_classes):
    """One-hot encodes a numpy array.

    Args:
        y: A np.array with EEG labels.
        num_classes: An integer for number of classes.
    
    Returns:
        A np.array with one-hot encoded labels.
    """
    return np.eye(num_classes, dtype='uint8')[y]


def load_data(batch_size, shuffle=True):
    """Loads data from numpy files.

    Args:
        batch_size: An integer for dataloader batch size.
        shuffle (default is True): A boolean for shuffling data samples.

    Returns:
        A tuple containing the train, val, and test dataloaders in the form
            (train_dataloader, val_dataloader, test_dataloader).
    """
    ## Loading the dataset
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_train_valid = np.load("data/person_train_valid.npy")
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    person_test = np.load("data/person_test.npy")

    ## Adjusting the labels so that 

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769
    
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(2115, 1000, replace=False)
    ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (X_train, X_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]

    X_train_prep, y_train_prep = train_data_prep(X_train, y_train, 2, 2, True)
    X_valid_prep, y_valid_prep = valid_data_prep(X_valid, y_valid)
    X_test_prep = test_data_prep(X_test)
    ## Random splitting and reshaping the data

    
    print('Shape of training set:', X_train_prep.shape)
    print('Shape of validation set:', X_valid_prep.shape)
    print('Shape of training labels:', y_train_prep.shape)
    print('Shape of validation labels:', y_valid_prep.shape)
    
    # Converting the labels to categorical variables for multiclass classification
    y_train_prep = to_categorical(y_train_prep, 4)
    y_valid_prep = to_categorical(y_valid_prep, 4)
    y_test = to_categorical(y_test, 4)
    
    print('Shape of training labels after categorical conversion:', y_train_prep.shape)
    print('Shape of validation labels after categorical conversion:', y_valid_prep.shape)
    print('Shape of test labels after categorical conversion:', y_test.shape)
    
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2])


    # Reshaping the training and validation dataset
    #x_train = np.swapaxes(x_train, 1, 3)
    #x_train = np.swapaxes(x_train, 1, 2)
    #x_valid = np.swapaxes(x_valid, 1, 3)
    #x_valid = np.swapaxes(x_valid, 1, 2)
    #x_test = np.swapaxes(x_test, 1, 3)
    #x_test = np.swapaxes(x_test, 1, 2)
    
    """
    print('Shape of training set after dimension reshaping:', x_train.shape)
    print('Shape of validation set after dimension reshaping:', x_valid.shape)
    print('Shape of test set after dimension reshaping:', x_test.shape)
    """
    
    train_iter = TensorDataset(torch.tensor(X_train_prep), torch.tensor(y_train_prep))
    val_iter = TensorDataset(torch.tensor(X_valid_prep), torch.tensor(y_valid_prep))
    test_iter = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    train_dataloader = DataLoader(dataset=train_iter, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(dataset=val_iter, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_iter, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader