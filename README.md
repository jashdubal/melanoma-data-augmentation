# melanoma-data-augmentation

## Install dependencies
```bash
pip install -r requirements.txt
```


## Part 1: Transform current data
```bash
python transform.py
```

## Part 2: Augment random images
Requires setting up huggingface account and downloading the dataset. Get a huggingface access token. Go to [huggingface.co/datasets/laion/laion400m](https://huggingface.co/datasets/laion/laion400m) and accept the terms.

Run:
```bash
huggingface-cli login
```
Put your token in the terminal when prompted. Then run:

```bash
python download_random.py
```

This will create a `random` directory in the `ISIC-images-split/train`, `ISIC-images-split/test`, and `ISIC-images-split/validation` directories.

## Part 3: Augment clear skin images
Available at: https://gallery.isic-archive.com/. Filter Lesion Images by: Benign - Other. Download all images.