import os
from PIL import Image

letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]

os.chdir("./individual_letters_grayscale")
for letter in letters:
    os.mkdir("./" + letter)

os.chdir("..")

for letter in letters:
    file_names = os.listdir("./individual_letters/" + letter)
    file_names.sort()
    for file_name in file_names:
        img = Image.open("./individual_letters/" + letter + "/" + file_name)
        img = img.convert("L")
        img.save("./individual_letters_grayscale/" + letter + "/" + file_name)
        print("converted img")
