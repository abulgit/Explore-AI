# AI-Powered Image Captioning

![Image Captioning Example](example_image.png)

## Project Description

This project explores the capabilities of Artificial Intelligence in the field of image captioning. Image captioning is the task of generating descriptive and meaningful captions for images automatically using deep learning techniques. The project involves building an end-to-end AI system that can analyze and understand the content of an image and then generate relevant captions.

The model architecture consists of a Convolutional Neural Network (CNN) for image feature extraction and a Long Short-Term Memory (LSTM) network for caption generation. The CNN extracts meaningful features from images, while the LSTM generates captions based on these features.

## Dataset

The project uses the [Pistachio Image Dataset]([https://cocodataset.org/](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)), a widely used dataset for object detection, segmentation, and captioning tasks. The dataset contains a large collection of images with corresponding captions.

## Getting Started

1. Clone the repository to your local machine:

```
git clone https://github.com/yourusername/ai-image-captioning.git
cd ai-image-captioning
```

2. Download the Pistachio Image Dataset and place the images in the `data/images/` directory. Create a `captions.txt` file in the `data/` directory with the corresponding image filenames and their captions.

3. Install the required dependencies (if any) listed in `requirements.txt`:

```
pip install -r requirements.txt
```

## Training the Model

To train the image captioning model, run the `train.py` script:

```
python train.py
```

The trained model will be saved as `trained_model.pth` in the project directory.

## Evaluating the Model

To evaluate the performance of the trained model, run the `evaluate.py` script:

```
python evaluate.py
```

The script will calculate the BLEU, METEOR, and CIDEr scores on the test dataset and display the results.

## Using the Model for Image Captioning

You can use the trained model to generate captions for your own images. Add the images to the `data/images/` directory and run the `generate_caption.py` script:

```
python generate_caption.py image_filename.jpg
```

Replace `image_filename.jpg` with the filename of the image for which you want to generate a caption.


## Results and Discussion

Summarize the model's performance and discuss any insights gained during the project. Mention the achieved BLEU, METEOR, and CIDEr scores and compare them with state-of-the-art methods if applicable.

## Acknowledgments

Acknowledge any resources, libraries, or tutorials used in the project.
