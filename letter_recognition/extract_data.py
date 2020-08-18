import imageio
import numpy as np
import pickle
import os

letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]

#data_set: 0 is training, 1 is validation, 2 is test
#returns a list of the data points
def getDataPoints(letter_index, data_set):
    data_points = []

    if data_set == 0:
        folder = "./individual_letters_contrast_and_grayscale_training"
    elif data_set == 1:
        folder = "./individual_letters_contrast_and_grayscale_validation"
    else:
        folder = "./individual_letters_contrast_and_grayscale_test"

    letter = letters[letter_index]
    file_names = os.listdir(folder + "/" + letter)
    file_names.sort()
    for file_name in file_names:
        img = imageio.imread(folder + "/" + letter + "/" + file_name)
        img_length = img.shape[0]
        pixel_array = np.zeros((img_length*img_length, 1))
        #iterate top-to-bottom, left-to-rights
        pixel_array_index = 0
        for i in range(img_length): #iterates over rows
            for j in range(img_length): #iterates over columns right to left
                k = (img_length - 1) - j #iterates over columns left to right
                pixel_array[pixel_array_index] = img[k,i] / 256
                pixel_array_index += 1

        target_array = np.zeros((len(letters), 1))
        target_array[letter_index] = 1
        data_points.append((pixel_array, target_array))
    return data_points


training_data = []
for l in range(len(letters)):
    training_data = training_data + getDataPoints(l, 0)

validation_data = []
for l in range(len(letters)):
    validation_data = validation_data + getDataPoints(l, 1)

test_data = []
for l in range(len(letters)):
    test_data = test_data + getDataPoints(l, 2)


with open("training_data.p", "wb") as f:
    pickle.dump(training_data, f)

with open("validation_data.p", "wb") as f:
    pickle.dump(validation_data, f)

with open("test_data.p", "wb") as f:
    pickle.dump(test_data, f)


