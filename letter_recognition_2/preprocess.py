#1. take data from jpg_images, convert to pngs, and split data
#2. sort by letter
#3. apply contrast, apply grayscale, generate additional training data by rotation, then sort into training, validation, and test data
#4. consolidate data into network-readable form
#5. output: 3 pickle files: training_data.p, validation_data.p, and test_data.p

from PIL import Image, ImageEnhance
import numpy as np
import os
import imageio
import shutil
import pickle

#BEGIN CONSTANT DEFINITONS

letter_length = 30
letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]
vertically_symmetric_letters = ["h", "i", "o", "s", "n"]
orientations = ["0", "90", "180", "270"]
num_slots_per_board = 25
num_board_arrangements = 25
num_pics_per_arrangement = 20
rotations = [5, -5, 10, -10]

#END CONSTANT DEFINITONS

#BEGIN HELPER METHODS

#orientation_index: 0 is normal, 1 is 90, 2 is 180, 3 is 270
def getImageNormallyOriented(img, orientation_index):
    if orientation_index == 0:
        return img
    elif orientation_index == 1:
        return img.rotate(90)
    elif orientation_index == 2:
        return img.rotate(180)
    elif orientation_index == 3:
        return img.rotate(270)

#returns a list of imgs in the 4 orientations
def getImageOrientations(img):
    return [img, img.rotate(270), img.rotate(180), img.rotate(90)]

#data_set: 0 is training, 1 is validation, 2 is test
#returns a list of the data points
def getDataPoints(letter_index, orientation_index, data_set):
    data_points = []

    if data_set == 0:
        folder = "./individual_letters_contrast_and_grayscale_training"
    elif data_set == 1:
        folder = "./individual_letters_contrast_and_grayscale_validation"
    else:
        folder = "./individual_letters_contrast_and_grayscale_test"

    letter = letters[letter_index]
    orientation = orientations[orientation_index]
    file_names = os.listdir(folder + "/" + letter + "_" + orientation)
    file_names.sort()
    for file_name in file_names:
        img = imageio.imread(folder + "/" + letter + "_" + orientation + "/" + file_name)
        pixel_array = np.zeros((letter_length*letter_length, 1))
        #iterate top-to-bottom, left-to-right
        pixel_array_index = 0
        for i in range(letter_length): #iterates over rows
            for j in range(letter_length): #iterates over columns right to left
                k = (letter_length - 1) - j #iterates over columns left to right
                pixel_array[pixel_array_index] = img[k,i] / 256
                pixel_array_index += 1

        target_array = np.zeros((len(letters) * len(orientations), 1))
        target_array[letter_index * 4 + orientation_index] = 1
        data_points.append((pixel_array, target_array))
    return data_points

#END HELPER METHODS

'''
#BEGIN CONVERTING TO PNGS, RESIZING, AND SPLITTING DATA

#includes rotating all images to upright
print("converting to pngs, resizing, and splitting data")
file_names = os.listdir("./jpg_images")
file_names.sort()

os.mkdir("./individual_letters")

letter_imgs = []
for i in range(len(file_names)):
    #save png file
    file_name = file_names[i]
    img = Image.open("./jpg_images/" + file_name)
    orientation_index = i % 4
    img = getImageNormallyOriented(img, orientation_index)

    img = img.resize((letter_length * 5, letter_length * 5))
    img.save("temp.png")

    img = imageio.imread("temp.png")
    print("    splitting img " + str(i+1) + " of " + str(len(file_names)))
    #img[y,x] refers to the pixel at row x, column y
    for j in range(5): #iterates over rows
        for k in range(5): #iterates over columns (right to left since imageio is weird)
            l = 4 - k #iterates over columns (left to right)
            letter_imgs.append(img[l*letter_length:(l+1)*letter_length, j*letter_length:(j+1)*letter_length, :])

j = 0
for letter_img in letter_imgs:
    imageio.imwrite("./individual_letters/" + str(j) + ".png", letter_img)
    print("    writing individual letter " + str(j+1) + " of " + str(len(letter_imgs)))
    j+=1
        
os.remove("temp.png")
#END CONVERTING TO PNGS, RESIZING, AND SPLITTING DATA

#BEGIN SORTING LETTERS
print("sorting letters")

num_lightings = len(os.listdir("./jpg_images")) // 500
num_individual_letters = len(os.listdir("./individual_letters"))

print("    creating directories")
os.chdir("./individual_letters")
for orientation in orientations:
    for letter in letters:
        os.mkdir("./" + letter + "_" + orientation)

os.chdir("..")

overall_pic_index = 0
letter_counts = [0] * len(letters)

for h in range(num_lightings):
    for i in range(num_board_arrangements):
        for j in range(num_pics_per_arrangement):
            for k in range(num_slots_per_board):
                letter_index = (k - i) % num_slots_per_board
                print("    sorting letter " + str(overall_pic_index + 1) + " of " + str(num_individual_letters))  
                if letter_index < len(letters):
                    letter = letters[letter_index]
                    img = Image.open("./individual_letters/" + str(overall_pic_index) + ".png")
                    image_orientations = getImageOrientations(img)
                    for l in range(len(orientations)):
                        image_orientations[l].save("./individual_letters/" + letter + "_" + orientations[l] + "/" + str(letter_counts[letter_index]) + ".png")

                    letter_counts[letter_index] += 1
                
                os.remove("./individual_letters/" + str(overall_pic_index) + ".png")
                overall_pic_index += 1

#END SORTING LETTERS

#BEGIN CONTRAST, GRAYSCALE, GENERATE TRAINING DATA
print("setting contrast and grayscale and generating additional training data")
folders = ["training", "validation", "test"]

print("    creating directories for training, validation, and test images")
for folder in folders:
    folder_path = "./individual_letters_contrast_and_grayscale_" + folder
    os.mkdir(folder_path)
    os.chdir(folder_path)
    for letter in letters:
        for orientation in orientations:
            os.mkdir(letter + "_" + orientation)
    
    os.chdir("..")

for letter in letters:
    for orientation in orientations:
        print("    processing files in " + letter + "_" + orientation)
        file_names = os.listdir("./individual_letters/" + letter + "_" + orientation)
        file_names.sort()
        count = 0
        for file_name in file_names:
            img = Image.open("./individual_letters/" + letter + "_" + orientation + "/" + file_name)
            enhancer = ImageEnhance.Contrast(img)
            factor = 5.0
            img = enhancer.enhance(factor)
            img = img.convert("L")
            img = img.rotate(270)
            
            if count % 5 == 0:
                img.save("./individual_letters_contrast_and_grayscale_test/" + letter + "_" + orientation + "/" + file_name)
            elif count % 5 == 1:
                img.save("./individual_letters_contrast_and_grayscale_validation/" + letter + "_" + orientation + "/" + file_name)
            else:
                img.save("./individual_letters_contrast_and_grayscale_training/" + letter + "_" + orientation + "/" + file_name)

            for rotation in rotations:
                rotated_img = img.rotate(rotation)
                img.save("./individual_letters_contrast_and_grayscale_training/" + letter + "_" + orientation + "/" + str(rotation) + "_rotation_" + file_name)

            count+=1

#shutil.rmtree("./individual_letters")
#END CONTRAST, GRAYSCALE, GENERATE TRAINING DATA
'''

#BEGIN EXTRACT DATA
print("extracting data")
training_data = []
validation_data = []
test_data = []
validation_data = []
for l in range(len(letters)):
    for k in range(len(orientations)):
        if not ((letters[l] in vertically_symmetric_letters) and (orientations[k] == "270" or orientations[k] == "180")):
            print("    extracting data from " + letters[l] + "_" + orientations[k])
            training_data[0:0] = getDataPoints(l, k, 0)
            validation_data[0:0] = getDataPoints(l, k, 1)
            test_data[0:0] = getDataPoints(l, k, 2)

print("    dumping")

with open("training_data.p", "wb") as f:
    pickle.dump(training_data, f)

with open("validation_data.p", "wb") as f:
    pickle.dump(validation_data, f)

with open("test_data.p", "wb") as f:
    pickle.dump(test_data, f)

#shutil.rmtree("./individual_letters_contrast_and_grayscale_training")
#shutil.rmtree("./individual_letters_contrast_and_grayscale_validation")
#shutil.rmtree("./individual_letters_contrast_and_grayscale_test")

#END EXTRACT DATA
