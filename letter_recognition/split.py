from PIL import Image
import numpy as np
import os
import imageio

if __name__ == "__main__":
    letter_imgs = []
    for i in range(125):
        img = imageio.imread("./png_images/" + str(i) + ".png")
        #img[y,x] refers to the pixel at row x, column y
        print("parsing img")
        letter_length = img.shape[0] // 5
        for j in range(5): #iterates over rows
            for k in range(5): #iterates over columns (right to left since imageio is weird)
                l = 4 - k #iterates over columns (left to right)
                letter_imgs.append(img[l*letter_length:(l+1)*letter_length, j*letter_length:(j+1)*letter_length, :])

    j = 0
    for letter_img in letter_imgs:
        imageio.imwrite("./individual_letters/" + str(j) + ".png", letter_img)
        print("wrote img " + str(j))
        j+=1
