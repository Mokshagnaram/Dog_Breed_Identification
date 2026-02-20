# Dog Breed Identification Using Transfer Learning

A production-ready mini project that predicts dog breeds from uploaded images using a transfer learning model based on **VGG19** (pretrained on ImageNet), a **Flask** backend, and an **HTML/CSS** frontend.

This project now uses **TensorFlow Datasets (TFDS)** to automatically download and prepare the **Stanford Dogs** dataset (`stanford_dogs`) on first run. If that download fails, it automatically falls back to **CIFAR-10** (dog class only) and creates synthetic breed folders for demo training.

## Project Structure

```text
DogBreedProject/
│
├── dataset/
├── model/
│   └── dog_breed_model.h5
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone or copy this project.
2. Move into the project directory:

```bash
cd DogBreedProject
```

3. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation (Automatic)

You do **not** need to manually download images.

When you run training, `train_model.py` will:
- Download `stanford_dogs` from TFDS on first run
- Cache raw TFDS data locally under `tfds_data/`
- Export images into this directory layout for `ImageDataGenerator`:

```text
dataset/
  train/
    breed_class_1/
    breed_class_2/
    ...
  test/
    breed_class_1/
    breed_class_2/
    ...
```

Fallback behavior:
- If Stanford Dogs is unavailable, it uses CIFAR-10 and keeps only dog images.
- It distributes those dog images into synthetic breed folders (`cifar_demo_breed_*`) so the transfer-learning pipeline still trains end-to-end.

## Training Instructions

Run:

```bash
python train_model.py
```

What training script does:
- Downloads and prepares Stanford Dogs dataset automatically (first run only)
- Loads image data with `ImageDataGenerator`
- Applies rescaling and augmentation to training data
- Builds transfer learning model using VGG19 (`include_top=False`)
- Freezes convolutional base
- Adds custom classifier head
- Trains with `EarlyStopping` and `ModelCheckpoint`
- Saves model to `model/dog_breed_model.h5`
- Saves training curves to `model/training_curves.png`

## Running Flask App

Start server:

```bash
python app.py
```

Open browser:
- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Example Usage

1. Train the model.
2. Open the web app home page.
3. Upload a dog image.
4. Click **Predict Breed**.
5. View predicted breed and confidence score.

## Notes

- If `model/dog_breed_model.h5` is missing, train the model first.
- Uploaded images are stored in `static/uploads/`.
- Compatible with Python 3.10+.
