# Feature Extraction

Code from my research internship at Purdue University (2022–23), where I worked on applications of graph neural networks to human-object interaction detection. These notebooks cover the preprocessing and feature extraction steps that came before the graph models.

## Notebooks

### `feature_extraction.ipynb`: region features from a pretrained detector

- Loads a pretrained Faster R-CNN with a ResNet-50 FPN backbone from `torchvision`
- `ImageFeatureExtractor` wraps the detector and returns the feature map from a chosen layer of the backbone
- `extract_image_deep_feature` runs an image through the network and collects the features for that layer
- `roi_align` uses `torchvision.ops.MultiScaleRoIAlign` to pool the feature map inside each bounding box, giving one feature vector per region

The result is a fixed-size feature vector for every object or person box in an image, which can then be used as node features in a graph.

### `a1.ipynb`: cleaning video feature tensors

The input is a `.pt` file of shape `[num_frames, 1024]`, holding a 1024-dimensional feature vector for every frame of a video. Some frames contain NaN values. The `replace_nan(path)` function:

1. Loads the tensor from the given path
2. Replaces each NaN with the mean of the non-NaN values in the same feature dimension across all frames
3. Returns the cleaned tensor

### `ResNet.ipynb`: ResNet-34 in Keras

A ResNet-34 built from scratch in TensorFlow/Keras, using identity blocks and convolutional (downsampling) blocks with batch normalisation and skip connections. It is set up for 32×32×3 inputs and 10 classes by default.

## Requirements

```bash
pip install torch torchvision torchsummary opencv-python numpy pandas tensorflow
```

The notebooks were run on Google Colab, so file paths point to `/content/...` and some cells use `google.colab.patches.cv2_imshow`. Change the paths to your own files when running them elsewhere.
