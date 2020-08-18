from PIL import Image, ImageEnhance
import os

letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]

rotations = [5, -5, 10, -10]

'''
os.chdir("./individual_letters_contrast_and_grayscale_training")
for letter in letters:
    os.mkdir("./" + letter)

os.chdir("..")

os.chdir("./individual_letters_contrast_and_grayscale_test")
for letter in letters:
    os.mkdir("./" + letter)

os.chdir("..")
'''

for letter in letters:
    file_names = os.listdir("./individual_letters/" + letter)
    file_names.sort()
    count = 0
    for file_name in file_names:
        img = Image.open("./individual_letters/" + letter + "/" + file_name)
        enhancer = ImageEnhance.Contrast(img)
        factor = 5.0
        img = enhancer.enhance(factor)
        img = img.convert("L")
        img = img.rotate(270)
        
        if count % 5 == 0:
            img.save("./individual_letters_contrast_and_grayscale_test/" + letter + "/" + file_name)
        elif count % 5 == 1:
            img.save("./individual_letters_contrast_and_grayscale_validation/" + letter + "/" + file_name)
        else:
            img.save("./individual_letters_contrast_and_grayscale_training/" + letter + "/" + file_name)

        for rotation in rotations:
            rotated_img = img.rotate(rotation)
            img.save("./individual_letters_contrast_and_grayscale_training/" + letter + "/" + str(rotation) + "_rotation_" + file_name)

        print("converted " + letter + " and made duplicates")

        count+=1
