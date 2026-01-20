The source code follows Mozilla Licence.

## Requirement library

```bash
pip install -r requirement.txt
```

## Usage

The final train code is `train5class.py`, `train5class_cv-all` is Stratified 5-Fold Cross-Validation version.

```python
python train5class.py
```

## Datasets

`datasets/dataset3class/`

We classify the datasets into training set and test set. The corresponding labels of images are classified as `Positive`, `NC` and `Weakly_positive`.

`datasets/H1N1/`
5 classification

`datasets/class_11/`
11 classification

"acc" is accuracy. "val" is validation.

test dataset's accuracy is 100%.

>  Test Loss: 0.0058, Test Accuracy: 1.0000, Test Accuracy Score: 1.0000
