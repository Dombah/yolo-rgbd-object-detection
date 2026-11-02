# Does depth help object detection? YOLOv12 on RGB vs. RGB-D

Most object detectors only see colour. Depth sensors add a fourth channel: how far each pixel is from the camera. This project measures how much that extra channel helps. It trains the same YOLOv12 detector three times on indoor scenes from **SUN RGB-D**: on RGB images only, on depth maps only, and on 4-channel **RGB-D** images. Everything else is kept identical.

**Result:** adding depth to RGB raised mAP@50-95 from **0.372 to 0.413** (+11% relative), and helped precision, recall and mAP@50 too.

Seminar paper for the *Image Processing and Computer Vision* course, MSc Computer Engineering, FERIT Osijek, 2025. The full paper (in Croatian) is in this repo.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ultralytics YOLOv12](https://img.shields.io/badge/Ultralytics-YOLOv12-111F68?style=flat)
![Google Colab](https://img.shields.io/badge/Colab-A100-F9AB00?style=flat&logo=googlecolab&logoColor=white)

<p align="center">
  <img src="models/RGB/val_batch0_pred.jpg" width="85%" alt="Detections on validation images">
</p>

---

## 📊 Results

Best epoch of each model, on the validation split (YOLOv12-s, 100 epochs, same settings for all three):

| Input | Channels | Precision | Recall | mAP@50 | mAP@50-95 |
|---|:-:|:-:|:-:|:-:|:-:|
| Depth only | 1 | 0.496 | 0.473 | 0.459 | 0.327 |
| RGB | 3 | 0.570 | 0.476 | 0.499 | 0.372 |
| **RGB-D** | **4** | **0.576** | **0.525** | **0.546** | **0.413** |

**What this means:**
- **Depth alone is surprisingly competitive.** The depth-only model reaches about 88% of the RGB model's mAP@50-95. That's without any colour or texture, from shape and layout alone.
- **Combining them works best.** RGB-D beats RGB on every metric. The biggest gain is in **recall** (+0.049): this suggests depth helps the model find objects that colour alone misses, plausibly overlapping furniture or objects that blend into the background.
- **Noisy depth can hurt.** In well-lit scenes where the depth map is noisy, RGB-D sometimes does worse than RGB. `vizualize.ipynb` finds images where the two models disagree most, for side-by-side inspection.

---

## ⚙️ Method

### 1. Data
SUN RGB-D has 10,335 indoor images, each with an aligned depth map and bounding-box annotations.

- **Class selection.** The annotations contain 1,068 distinct object names, most too rare to learn from. I kept the **45 most frequent classes**, from chairs (24,599 instances) down to books (208). Images with none of these classes were dropped.
- **Format.** Annotations were converted from SUN RGB-D's JSON to YOLO label files.
- **Split.** 80/20 train/validation, using **multi-label stratified splitting** so that each class is proportionally represented in both sets.
- **Inputs.** Each image was exported three ways: RGB (`.jpg`), depth (`.png`) and a 4-channel RGB-D stack (`.tiff`).

### 2. Making YOLO accept 4 channels
Ultralytics YOLO is built for 3-channel images. To train on RGB-D I modified the library:
- image loading reads `.tiff` with `tifffile` instead of `cv2.imread`, which drops extra channels
- the model config sets `ch: 4`, so the first convolution takes 4 input channels

### 3. A fair comparison
Training on RGB-D forced some settings, and I applied them to **all three** models so the comparison stays fair:
- **No augmentation.** The library's colour and geometric augmentations only support 3 channels and would silently drop depth. Mosaic still applies, and is turned off for the last 20 epochs.
- **No COCO pretraining.** Pretrained weights fix the first layer at 3 channels, so every model trains from scratch.

All three models are YOLOv12-s, trained for 100 epochs at batch size 64 and 640×640 on a Colab A100. Each peaked between epochs 80 and 90.

---

## 📁 Repository

```
preprocessing.ipynb    class selection, JSON → YOLO labels, stratified split, RGB / depth / RGB-D export
main.ipynb             training on Colab (RGB, depth, RGB-D)
vizualize.ipynb        dataset statistics, per-image model comparison, disagreement examples
models_yaml/           YOLOv12-s configs (RGB-D sets ch: 4)
models_data/           dataset configs per input type
models/RGB, RGBD/      training curves, confusion matrices, sample predictions, best/last weights
classes.txt            the 45 selected classes
Analiza doprinosa dubinskih informacija u detekciji objekata.pdf   full paper (Croatian)
```

---

## ⚠️ Limitations

- **Single run per model.** The differences haven't been checked across random seeds.
- **No separate test set.** Results are reported on the validation split that was also used to pick the best epoch.
- **Lower absolute scores than usual.** Because augmentation and pretraining were off for all three models, the numbers are lower than a tuned detector would reach. The point is the *relative* comparison.
- **Naive fusion.** Depth is stacked as a 4th input channel. Dedicated fusion architectures, such as a separate depth branch, might gain more.
