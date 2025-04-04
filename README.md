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
Idea 1: Use thispersondoesnotexist.com and generate 3 images (left cheek, right cheek, chin)
Idea 2: Use makeup dataset: https://www.kaggle.com/datasets/tobyclh/microglam/data