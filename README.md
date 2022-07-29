# Image clustering using perceptual hashing

This is a tool for clustering images using perceptual hashing.

## Setting up

You need git and Python 3 installed.

```commandline
git clone https://github.com/peterlevi/image-clustering.git
cd image-clustering
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Put images to cluster inside an `data/images` subfolder of the folder where you are running the
script.
Then run with:

```commandline
./cluster.py
```

The output clusters will be written into an `data/output` subfolder (will be auto-created if
necessary).

## Implementation

ImageHash (https://github.com/JohannesBuchner/imagehash) provides the implementation of
phash, dhash, average hash, wavelet hash.

Hashing takes 0.5-1 second per image, and is run for each of the hash algorithms configured in
`ALGORITHMS`. Multiprocessing is used to speed up hashing.

After we compute the image hashes with ImageHash, we iterate a range of thresholds.
For each we compare the hash differences between pairs of images against the threshold, and for
the pairs which are below the threshold (indicating similar images at this threshold),
we compute the connected graph components and call those the "clusters" at that threshold.

These are then written to the output folder (using symlinks to the images), along with all
unclustered images.

## Interpreting results

In the output folder, you'll see several subfolders for different image hashing algorithms, run wih
different parameters. Some may produce better results than others for your particular image set.

Within each of these folders, you will see subfolders for every threshold level.
The number at the end indicates the number of ungrouped images. If too many images are ungrouped,
this threshold may be too low, and failing to group even slightly different images.
Too few ungrouped images, on the other hand, probably means the threshold is too high and dissimilar
images would have ended up being grouped together.

In each such folder, there will be separate folders for every cluster of images, and
an `unclustered` folder. The numbers at the end indicate the number of images in the group. Groups
will be sorted by size (`001` being the largest).

Every group folder contains symlinks to the `data` folder where the original images reside.

A photo viewer with good browsing capabilities can then be used to explore the clusters visually -
for Linux a good one for this purpose is my own
[Ojo Image Viewer](https://github.com/peterlevi/ojo/).


