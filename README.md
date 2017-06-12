TensorFlow project for recognizing captcha by multi-label classification.

In this demo project, the captcha contains 6 characters, including `0~9, a~z, A~Z`. The image is of size `60x150x1`.

Step 1: Download the data in npy: download the data from the link below

```
https://drive.google.com/open?id=0BxG1qYaVrovrVDFyZGxqRjJPTms
```

Step 2: unzip the downloaded zip file to `./data/`, so that the directory should be like this:

```
-- data --
        |---- train_data.npy
        |---- test_data.npy
        |---- train_label.npy
        |---- test_label.npy
```

Step 3: run python main.py
