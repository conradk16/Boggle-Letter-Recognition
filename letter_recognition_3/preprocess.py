from PIL import Image, ImageEnhance
import numpy as np
import os
import imageio
import pickle
import glob
import network2
import shutil
import re
import random

#save jpgs in a directory called "jpg_images" and label each image "<n>.jpg", where is the number of the image
#save labels in a text file with a space separated list of (<lowercaseletterOrientation> elements,where orientations = 0,1,2,3) on each line . One line corresponds to one image. Label file should be called "labels.txt"

letter_length = 30
letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]
vertically_symmetric_letters = ["h", "i", "o", "s", "n"]
orientations = [0,1,2,3]
letters_and_orientations = []
for letter in letters:
    for orientation in orientations:
        letters_and_orientations.append(letter + str(orientation))

numbers = list(range(len(letters_and_orientations)))
letters_to_indices = dict(zip(letters_and_orientations, numbers))

#HELPER METHODS
#orientation_index: 0 is normal, 1 is 90, 2 is 180, 3 is 270
#img is a PIL image
def getImageNormallyOriented(img, orientation_index):
    return img.rotate(orientation_index * 90)

#returns a list of imgs in the 4 orientations
def getImageOrientations(img):
    return [img, img.rotate(270), img.rotate(180), img.rotate(90)]
    
def save1dNumpyArrayAsPNG(path, array):
    pixel_array_index = 0
    letter_img = np.zeros((letter_length, letter_length))
    for m in range(letter_length): #iterates over rows
        for j in range(letter_length): #iterates over columns right to left
            k = (letter_length - 1) - j #iterates over columns left to right
            letter_img[k,m] = int(array[pixel_array_index] * 256)
            pixel_array_index += 1

    imageio.imwrite(path, letter_img)

def saveDataPointAsPNG(folder, img_number, data_point):
    pixel_array, target_array = data_point
    label = letters_and_orientations[np.argmax(target_array)]
    save1dNumpyArrayAsPNG(folder + "/" + str(img_number) + label + ".png", pixel_array)

def get1dNumpyArrayFromSquarePNG(path, img_length):
    img = imageio.imread(path)
    pixel_array = np.zeros((img_length*img_length, 1))
    #iterate top-to-bottom, left-to-right
    pixel_array_index = 0
    for i in range(img_length): #iterates over rows
        for j in range(img_length): #iterates over columns right to left
            k = (img_length - 1) - j #iterates over columns left to right
            pixel_array[pixel_array_index] = img[k,i] / 256
            pixel_array_index += 1
    return pixel_array

def getLetterIndexFromLabel(label):
    index = letters_to_indices[label]
    return index // 4

def getOrientationFromLabel(label):
    index = letters_to_indices[label]
    return index % 4    

#returns a list of 4 2-tuples, one data point for each orientation
def getFourDataPointsFromLetterImage(path, label):
    data = []
    img = Image.open(path)
    img_orientation = getOrientationFromLabel(label)
    normal_image = getImageNormallyOriented(img, img_orientation)
    images = getImageOrientations(normal_image)
    letter_index = getLetterIndexFromLabel(label)

    for i in range(len(images)):
        img = images[i]
        img.rotate(270)
        img.save("temp.png", quality=95)
        target_index = letter_index * 4 + i
        pixel_array = get1dNumpyArrayFromSquarePNG("temp.png", letter_length)

        target_array = np.zeros((len(letters_and_orientations), 1))
        target_array[target_index] = 1
        data.append((pixel_array, target_array))
    os.remove("temp.png")
    return data

def generateAdditionalTrainingData(data_point, rotations):
    data = []
    pixel_array, target_array = data_point
    save1dNumpyArrayAsPNG("temp.png", pixel_array)
    img = Image.open("temp.png")
    for rotation in rotations:
        rotated_img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)
        rotated_img.save("temp.png", quality=95)
        pixel_array = get1dNumpyArrayFromSquarePNG("temp.png", letter_length)
        data.append((pixel_array, target_array))
    
    os.remove("temp.png")
    return data

#create directory for letters
os.mkdir("individual_letters")

#convert to pngs, split, and write to "individual_letters" directory
i = 0
images = glob.glob("jpg_images/*.jpg")
images.sort(key=lambda f: int(re.sub('\D', '', f)))
for m in range(len(images)):
    print("splitting image " + str(m+1) + " of " + str(len(images)))
    img = images[m]
    img = Image.open(img)
    img = img.resize((150,150))
    img.save("temp.png")
    
    img = imageio.imread("temp.png")
    letter_length = img.shape[0] // 5
    letter_imgs = []

    #split
    for j in range(5): #iterates over rows
        for k in range(5): #iterates over columns (right to left since imageio is weird)
            l = 4 - k #iterates over columns (left to right)
            letter_imgs.append(img[l*letter_length:(l+1)*letter_length, j*letter_length:(j+1)*letter_length, :])

    #write images
    for letter_img in letter_imgs:
        imageio.imwrite("./individual_letters/" + str(i) + ".png", letter_img)
        i+=1

os.remove("temp.png")

#add gray and contrast
file_names = glob.glob("individual_letters/*.png")
file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
for i in range(len(file_names)):
    print("adding grayscale and contrast to image " + str(i+1) + " of " + str(len(file_names)))
    file_name = file_names[i]
    img = Image.open(file_name)
    enhancer = ImageEnhance.Contrast(img)
    factor = 5.0
    img = enhancer.enhance(factor)
    img = img.convert("L")
    img = img.rotate(270)

    img.save(file_name, quality=95)

#save data as list of lists, one list for each letter. Each element of a list is a 2-tuple of 1d numpy arrays representing a point.
#for each individual letter, create 4 data points, one for each orientation

with open("labels.txt") as f:
    text = f.read()

labels = re.split(r'\s\s*', text)

data = []
#create sublists
for i in range(len(letters)):
    sublist = []
    data.append(sublist)

#get data and remove excess data points
file_names = glob.glob("individual_letters/*.png")
file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
for m, label in zip(list(range(len(file_names))), labels):
    print("extracting data from letter image " + str(m+1) + " of " + str(len(file_names)))
    file_name = file_names[m]
    letter_index = getLetterIndexFromLabel(label)
    fourDataPoints = getFourDataPointsFromLetterImage(file_name, label)
    if letters[letter_index] in vertically_symmetric_letters:
        data[letter_index][0:0] = fourDataPoints[:2]
    elif letters[letter_index] == "v":
        continue
    else:
        data[letter_index][0:0] = fourDataPoints
        
shutil.rmtree("individual_letters")

#remove excess data points
print("removing excess data points")
smallest_letter_list_size = len(data[0])
lowest_freq_letter = "default"
for i in range(len(data)):
    letter_list = data[i]
    size = len(letter_list) * 2 if letters[i] in vertically_symmetric_letters else len(letter_list)
    if len(letter_list) < smallest_letter_list_size and letters[i] != "v":
        smallest_letter_list_size = len(letter_list)
        lowest_freq_letter = letters[i]

print("lowest frequency letter: " + lowest_freq_letter)
print("number of occurrences (including rotations) = " + str(smallest_letter_list_size))


for i in range(len(data)):
    letter_list = data[i]
    data[i] = letter_list[:smallest_letter_list_size]

#create training, validation, and test pickle files
training_data = []
validation_data = []
test_data = []

for m in range(len(data)):
    letter_list = data[m]
    print("sorting into training, validation, test, and generating add'l training data from " + letters[m] + " list")
    random.shuffle(letter_list)
    for i in range(len(letter_list)):
        data_point = letter_list[i]
        if i % 5 == 0:
            validation_data.append(data_point)
        else:
            additional_training_data_points = generateAdditionalTrainingData(data_point, [-1.25, -2.5, -3.75, -5, -6.25, -7.5, 1.25, 2.5, 3.75, 5, 6.25, 7.5])
            training_data.append(data_point)
            training_data[0:0] = additional_training_data_points

print("dumping")
with open("training_data.p", "wb") as f:
    pickle.dump(training_data, f)

with open("validation_data.p", "wb") as f:
    pickle.dump(validation_data, f)

with open("test_data.p", "wb") as f:
    pickle.dump(test_data, f)

'''
#recover images and take a peekie poo
folders = ["training_images", "validation_images", "test_images"]
data = []
pickle_files = ["training_data.p", "validation_data.p", "test_data.p"]
num_sets = 3

for i in range(num_sets):
    os.mkdir(folders[i])
    with open(pickle_files[i], "rb") as f:
        data.append(pickle.load(f))

    for j in range(len(data[i])):
        data_point = data[i][j]
        saveDataPointAsPNG(folders[i], j, data_point)
'''
