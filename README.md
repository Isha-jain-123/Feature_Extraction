The assignment a1 does the following:
The shape of the .pt will is of type [Num_frames, 1024]. This array is nothing but a 1024-dimensional feature vector for each of the frame of a video.This array contains NaN values at some of the frames. We need to remove these NaN values.I have written a python function :
1. Which takes as input the location of the file.
2. The function removes NaN values by replacing them with the average of all the non NaN values present at that location in all the frames.
3. The function returns the filtered tensor without any NaN values.

The feature extraction file extracts features from a specific layer by utilising a Region Proposal Network along with Region of Interest (RoI) Pooling
