# Image clustering using perceptual hashing

This is a tool for clustering images using perceptual hashing.

## Running
Put images to cluster inside an `images` subfolder to the folder where you are running the script.
The output clusters will be written into an `output` subfolder (will be auto-created if necessary).


## Implementation
ImageHash (https://github.com/JohannesBuchner/imagehash) provides the implementation of 
phash, dhash, and average hash.

Hashing takes 0.5-1 second per image, and is run for each of the hash algorithms configured in 
ALGORITHMS.

After we compute the image hashes with ImageHash, we iterate a range of thresholds. 
For each we compare the hash differences between pairs of images against the threshold, and for 
the pairs which are below the threshold (indicating similar images at this threshold), 
we compute the connected graph components and call those the "clusters" at that threshold.

These are then written to the output folder (using symlinks to the images), along with all
unclustered images.

A photo viewer with good browsing capabilities can then be used to explore the clusters visually - 
for Linux the best one for this very purpose is my own Ojo Image Viewer (https://github.com/peterlevi/ojo/).
