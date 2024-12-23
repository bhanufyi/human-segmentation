# Semantic Segmentation Using PyTorch and Segmentation Models PyTorch (SMP)

This project implements a semantic segmentation pipeline using a `UNet` architecture from the `segmentation_models_pytorch` library. The pipeline includes data preprocessing, augmentation, dataset preparation, training, and inference.

---

## Requirements

### Libraries and Frameworks

The code requires the following Python libraries:

- `torch`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `albumentations`
- `segmentation-models-pytorch`
- `tqdm`

### Installation

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

The dataset is assumed to have:

1. A CSV file (`train.csv`) with columns:
    - `images`: Paths to image files.
    - `masks`: Paths to corresponding mask files.
2. Images and masks stored in the specified `DATA_DIR`.

Example structure:

```
DATA_DIR/
├── train.csv
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...

```

---

## How to Run the Project

### Step 1: Configuration

Set the necessary configuration parameters in the code:

- `CSV_FILE`: Path to the CSV file.
- `DATA_DIR`: Base directory of the dataset.
- `DEVICE`: Set automatically based on CUDA availability.
- Other hyperparameters: `EPOCHS`, `LR`, `IMAGE_SIZE`, `BATCH_SIZE`.

### Step 2: Data Augmentation

Custom augmentations are defined using `albumentations`. This ensures:

- Resizing images and masks to the desired size.
- Adding transformations like flipping for better generalization.

### Step 3: Training

- The segmentation model is based on a UNet architecture with an EfficientNet encoder.
- Training uses a combination of Dice Loss and Binary Cross Entropy Loss.
- The training and validation datasets are split 80-20 using `train_test_split`.

### Step 4: Inference

Use the `inference()` function to visualize predictions:

```python
inference(<index>)

```

This loads the best model saved during training and visualizes the input image, ground truth mask, and predicted mask.

---

## Code Breakdown

### 1. **Data Augmentation**

Defined with `albumentations`:

- `get_train_augs`: Resizing, horizontal, and vertical flipping.
- `get_valid_augs`: Only resizing for validation.

### 2. **Custom Dataset**

The `SegmentationDataset` class extends `torch.utils.data.Dataset` to:

- Load and preprocess images and masks.
- Apply augmentations.
- Convert data to tensors for model consumption.

### 3. **Model Architecture**

The segmentation model uses UNet with EfficientNet-B0 as the encoder:

- Input channels: 3 (RGB).
- Output channels: 1 (binary segmentation).
- Pretrained weights: `imagenet`.

### 4. **Training and Validation**

The `train_fn` and `eval_fn` functions handle the training and evaluation loops, respectively. They:

- Perform forward passes.
- Compute losses.
- Update model parameters during training.

### 5. **Saving and Loading Models**

The model saves the best checkpoint (`best_model.pth`) based on validation loss. It is used during inference.

### 6. **Inference**

The `inference()` function:

- Loads the best model.
- Processes a validation set image.
- Visualizes the input image, ground truth, and predicted mask.

---

## Results

Visualize the results using the helper function in `helper.py`. Example:

```python
helper.show_image(image, mask, pred_mask)

```

---

## Helper File

The `helper.py` file should include utility functions like:

- `show_image`: To visualize images, masks, and predictions.

Example structure:

```python
def show_image(image, ground_truth, prediction=None):
    ...

```

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgments

- [Coursera Deep Learning with PyTorch Image Segmentation](https://www.coursera.org/projects/deep-learning-with-pytorch-image-segmentation)
- [Segmentation Models PyTorch Documentation](https://smp.readthedocs.io/en/latest/)
- [Albumentations Documentation](https://albumentations.ai/docs/)

Feel free to reach out for questions or collaborations! 😊
