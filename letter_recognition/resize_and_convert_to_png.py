from PIL import Image
import numpy as np
import os

if __name__ == "__main__":
    file_names = os.listdir("./jpg_images")
    file_names.sort()
    j = 0
    for file_name in file_names:
        img = Image.open("./jpg_images/" + file_name)
        img = img.resize((150,150))
        img.save("./png_images/" + str(j) + ".png")
        j += 1
        print("saved img")
