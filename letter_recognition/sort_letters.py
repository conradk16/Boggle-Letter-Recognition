import os
from shutil import copyfile

letters = ["a", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "qu", "r", "s", "t", "u", "v", "w", "y"]

os.chdir("./individual_letters")
for letter in letters:
    os.mkdir("./" + letter)

os.chdir("..")

num_slots_per_board = 25
num_board_arrangements = 25
num_pics_per_arrangement = 5
overall_pic_index = 0
letter_counts = [0] * len(letters)

for i in range(num_board_arrangements):
    for j in range(num_pics_per_arrangement):
        for k in range(num_slots_per_board):
            letter_index = (k - i) % num_slots_per_board
            if letter_index < len(letters):
                letter = letters[letter_index]
                os.rename("./individual_letters/" + str(overall_pic_index) + ".png", "./individual_letters/" + letter + "/" + str(letter_counts[letter_index]) + ".png")
                letter_counts[letter_index] += 1
                print("moved " + letter + ". overall_pic_index = " + str(overall_pic_index))
            else:
                os.remove("./individual_letters/" + str(overall_pic_index) + ".png")
                print("deleted a blank" + ". overall_pic_index = " + str(overall_pic_index))
            overall_pic_index += 1
            
        
