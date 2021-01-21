from PIL import Image, ImageEnhance
import numpy as np
import os
import imageio
import pickle
import glob
import network2
import shutil
import re

#save network in a file called "saved_network.p"
#save jpgs in a directory called "jpg_test_images" and label each image "<n>.jpg", where n is the number of the image
#save labels in a text file with a space separated list of (lowercase) letters on each line. One line corresponds to one image. Label file should be called "labels.txt"

#helper method for debugging
def saveNumpyArrayAsPNG(path, array):
    pixel_array_index = 0
    letter_img = np.zeros((letter_length, letter_length))
    for m in range(letter_length): #iterates over rows
        for j in range(letter_length): #iterates over columns right to left
            k = (letter_length - 1) - j #iterates over columns left to right
            letter_img[k,m] = array[pixel_array_index] * 256
            pixel_array_index += 1

    imageio.imwrite(path, letter_img)

letter_length = 30
letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]
orientations = ["0", "90", "180", "270"]

#create directory for letters
os.mkdir("individual_letters")

#convert to pngs, split, and write to "individual_letters" directory
i = 0
test_images = glob.glob("jpg_test_images/*.jpg")
test_images.sort(key=lambda f: int(re.sub('\D', '', f)))
for img in test_images:
    img = Image.open(img)
    img = img.resize((150,150))
    img.save("test_img.png")
    
    img = imageio.imread("test_img.png")
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

os.remove("test_img.png")

#add gray and contrast
file_names = glob.glob("individual_letters/*.png")
file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
for file_name in file_names:
    img = Image.open(file_name)
    enhancer = ImageEnhance.Contrast(img)
    factor = 5.0
    img = enhancer.enhance(factor)
    img = img.convert("L")
    img = img.rotate(270)

    img.save(file_name)

#save data as list of 1d numpy arrays, one element for each individual letter
data = []
file_names = glob.glob("individual_letters/*.png")
file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
for file_name in file_names:
    img = imageio.imread(file_name)
    pixel_array = np.zeros((letter_length*letter_length, 1))
    #iterate top-to-bottom, left-to-right
    pixel_array_index = 0
    for i in range(letter_length): #iterates over rows
        for j in range(letter_length): #iterates over columns right to left
            k = (letter_length - 1) - j #iterates over columns left to right
            pixel_array[pixel_array_index] = img[k,i] / 256
            pixel_array_index += 1

    data.append(pixel_array)    

#test network
with open("labels.txt") as f:
    text = f.read()

labels = re.split(r'\s\s*', text)

with open("saved_network.p", "rb") as f:
    network = pickle.load(f)

correct = 0
tested = 0
for i in range(len(data)):
    label = labels[i]
    if label[0] == 'q':
        label = label[:2]
    else:
        label = label[:1]

    pixel_array = data[i]
    
    output_index = np.argmax(network.feedforward(pixel_array))
    network_output = letters[output_index // 4]
    print("network says: it's: " + network_output)
    print("actual letter: " + label)
    if network_output == label:
        correct += 1
    tested += 1

print(str(correct) + "/" + str(tested))

shutil.rmtree("individual_letters")
