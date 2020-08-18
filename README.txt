Letter recognition program for Boggle.

Directory: letter_recognition

This is the first iteration of the project, and it deals only with letters in the normal orientation. Different preprocessing tasks are each in their own files. This iteration of the project was more of a test to see if the desired results could be achieved in a minimal setting. To reduce data collection difficulties, the letters k, b, j, z, and x were not considered. Hence the network has only 21 output nodes. Training data was taken in a systematic fashion: 5 pictures taken for each board setup x 25 board setups = 125 images. Each subsequent board setup is created by taking the previous board setup with all letters shifted one to the right. Letters at the right end of a row are moved to the next row at the left.

Directory: letter_recognition_2

This is the second iteration of the project, and it deals with letters in all 4 orientations. The network has 84 output nodes, one for each letter-orientation pair for the subset of letters not including k, b, j, z, and x. Preprocessing is done all with one python program. Training data was taken in a systematic fashion: 20 pictures were taken for each board setup x 25 board setups x 2 lightings = 1000 images. The 20 pictures were taken at angles 0, 90, 180, 270, 0, 90, .... The board setups were the same as for iteration 1 of the project.

To use, take 1000 images in the order described and place in folder called "jpg_images". Then run preprocess.py, followed by learn.py

Directory: letter_recognition_3

This is the third iteration of the project, and it is an expansion of the second iteration for randomly taken training images. To run, place training images in a folder called "jpg_images" and place labels in "labels.txt". Labels should be labelings of the images, one per line. One label is of the form <lowercase_letter><orientation>, where orientation is a number 0-3. This iteration of the project removes the letter v from training, since its low frequency was limiting the amount of training data.
