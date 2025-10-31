# TorchVision Datasets and Recommended Models

This directory contains downloaded TorchVision datasets and their compatible models for the ACT verification framework.

## Dataset-Model Mapping Table

### Classification Datasets

| Dataset          | Category       | Input Size    | Classes | Recommended Models                                                                        |
|------------------|----------------|---------------|---------|-------------------------------------------------------------------------------------------|
| MNIST            | Classification | 1×28×28       | 10      | simple_cnn, lenet5, resnet18, efficientnet_b0                                            |
| FashionMNIST     | Classification | 1×28×28       | 10      | simple_cnn, lenet5, resnet18, efficientnet_b0                                            |
| KMNIST           | Classification | 1×28×28       | 10      | simple_cnn, lenet5, resnet18                                                             |
| QMNIST           | Classification | 1×28×28       | 10      | simple_cnn, lenet5, resnet18                                                             |
| EMNIST           | Classification | 1×28×28       | 62      | simple_cnn, resnet18                                                                     |
| CIFAR10          | Classification | 3×32×32       | 10      | resnet18, resnet34, resnet50, vgg16, mobilenet_v2, efficientnet_b0                       |
| CIFAR100         | Classification | 3×32×32       | 100     | resnet18, resnet34, resnet50, vgg16, mobilenet_v2, efficientnet_b0                       |
| STL10            | Classification | 3×96×96       | 10      | resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b0                              |
| SVHN             | Classification | 3×32×32       | 10      | resnet18, resnet34, vgg16, mobilenet_v2, efficientnet_b0                                 |
| Caltech101       | Classification | 3×224×224     | 101     | resnet50, efficientnet_b0, vit_b_16, convnext_tiny                                       |
| Caltech256       | Classification | 3×224×224     | 257     | resnet50, resnet101, efficientnet_b0, vit_b_16                                           |
| Flowers102       | Classification | 3×224×224     | 102     | resnet50, efficientnet_b0, vit_b_16, convnext_tiny                                       |
| Food101          | Classification | 3×224×224     | 101     | resnet50, resnet101, efficientnet_b1, vit_b_16                                           |
| OxfordIIITPet    | Classification | 3×224×224     | 37      | resnet50, efficientnet_b0, mobilenet_v2, vit_b_16                                        |
| StanfordCars     | Classification | 3×224×224     | 196     | resnet50, resnet101, efficientnet_b3, vit_b_16, convnext_small                           |
| FGVCAircraft     | Classification | 3×224×224     | 100     | resnet50, resnet101, efficientnet_b3, vit_b_16                                           |
| EuroSAT          | Classification | 3×64×64       | 10      | resnet18, resnet50, efficientnet_b0, vit_b_16                                            |
| SUN397           | Classification | 3×224×224     | 397     | resnet50, resnet101, vgg16, densenet161                                                  |
| Country211       | Classification | 3×224×224     | 211     | resnet50, efficientnet_b0, vit_b_16                                                      |
| Omniglot         | Classification | 1×105×105     | 1623    | simple_cnn, resnet18                                                                     |
| PCAM             | Classification | 3×96×96       | 2       | resnet18, resnet50, efficientnet_b0                                                      |
| INaturalist      | Classification | 3×224×224     | 8142    | resnet50, resnet101, efficientnet_b3, vit_b_16                                           |
| CelebA           | Classification | 3×224×224     | 40      | resnet34, resnet50, mobilenet_v2, efficientnet_b0                                        |
| LFWPeople        | Classification | 3×224×224     | 5749    | resnet34, resnet50, mobilenet_v2, efficientnet_b0                                        |
| LFWPairs         | Classification | 3×224×224     | 2       | resnet34, resnet50, mobilenet_v2                                                         |
| ImageNet         | Classification | 3×224×224     | 1000    | resnet18/34/50/101/152, vgg11/13/16/19, efficientnet_b0-b7, convnext_*, vit_*, swin_*   |
| Places365        | Classification | 3×224×224     | 365     | resnet50, resnet152, vgg16, densenet161                                                  |

### Detection Datasets

| Dataset        | Category  | Input Size | Classes | Recommended Models                                                                                  |
|----------------|-----------|------------|---------|-----------------------------------------------------------------------------------------------------|
| CocoDetection  | Detection | Variable   | 80      | fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn, ssd300_vgg16   |
| VOCDetection   | Detection | Variable   | 20      | fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn, ssd300_vgg16   |
| WIDERFace      | Detection | Variable   | 1       | fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn                                                     |

### Segmentation Datasets

| Dataset          | Category      | Input Size    | Classes | Recommended Models                                                                            |
|------------------|---------------|---------------|---------|-----------------------------------------------------------------------------------------------|
| VOCSegmentation  | Segmentation  | Variable      | 21      | fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, lraspp_mobilenet_v3    |
| Cityscapes       | Segmentation  | 3×1024×2048   | 19      | fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101                         |
| SBDataset        | Segmentation  | Variable      | 20      | fcn_resnet50, deeplabv3_resnet50                                                              |

### Video Datasets

| Dataset   | Category | Input Size | Classes | Recommended Models                                                    |
|-----------|----------|------------|---------|-----------------------------------------------------------------------|
| Kinetics  | Video    | T×H×W      | 400     | r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v1_b, mvit_v2_s, swin3d_*     |
| HMDB51    | Video    | T×H×W      | 51      | r3d_18, mc3_18, r2plus1d_18                                           |
| UCF101    | Video    | T×H×W      | 101     | r3d_18, mc3_18, r2plus1d_18, s3d                                      |

### Optical Flow Datasets

| Dataset        | Category      | Input Size | Classes | Recommended Models      |
|----------------|---------------|------------|---------|-------------------------|
| FlyingChairs   | Optical Flow  | Variable   | -       | raft_large, raft_small  |
| FlyingThings3D | Optical Flow  | Variable   | -       | raft_large, raft_small  |
| Sintel         | Optical Flow  | Variable   | -       | raft_large, raft_small  |
| KittiFlow      | Optical Flow  | Variable   | -       | raft_large, raft_small  |

## Summary Statistics

- **Total Datasets**: 40
- **Total Unique Models**: 63
- **Total Dataset-Model Pairs**: 189
- **Categories**: 5 (Classification, Detection, Segmentation, Video, Optical Flow)
- **Custom Models**: 2 (SimpleCNN, LeNet5)
- **Standard TorchVision Models**: 61

## Model Families

### Custom Models (1-channel input)
- **simple_cnn**: Simple CNN for grayscale images (28×28)
- **lenet5**: Classic LeNet-5 architecture

### ResNet Family
- resnet18, resnet34, resnet50, resnet101, resnet152

### VGG Family
- vgg11, vgg13, vgg16, vgg19

### EfficientNet Family
- efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

### Vision Transformers
- vit_b_16, vit_b_32, vit_l_16, vit_l_32

### Swin Transformers
- swin_t, swin_s, swin_b
- swin3d_t, swin3d_s, swin3d_b

### ConvNeXt Family
- convnext_tiny, convnext_small, convnext_base, convnext_large

### MobileNet Family
- mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

### DenseNet Family
- densenet121, densenet161, densenet169, densenet201

### Detection Models
- fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
- retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
- ssd300_vgg16, ssdlite320_mobilenet_v3_large
- fcos_resnet50_fpn

### Segmentation Models
- fcn_resnet50, fcn_resnet101
- deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
- lraspp_mobilenet_v3_large

### Video Models
- r3d_18, mc3_18, r2plus1d_18
- s3d
- mvit_v1_b, mvit_v2_s

### Optical Flow Models
- raft_large, raft_small

## Usage

To download a dataset-model pair:
```bash
python act/front_end/torchvision/data_model_mapping.py --download DATASET MODEL [--split train|test|both]
```

To list downloaded pairs:
```bash
python act/front_end/torchvision/data_model_mapping.py --list-downloads
```

To load a downloaded pair:
```bash
python act/front_end/torchvision/data_model_mapping.py --load-torchvision DATASET MODEL
```

## Directory Structure

Each downloaded dataset-model pair follows this structure:
```
<DATASET>/
├── raw/                 # Raw dataset files
├── models/             # Model architecture definitions
│   └── <model>.py
└── info.json          # Metadata (splits, preprocessing, classes, etc.)
```

## Notes

- **Preprocessing**: Many datasets require preprocessing (resize, grayscale→RGB, normalization)
- **Custom Models**: SimpleCNN and LeNet5 are custom implementations for grayscale inputs
- **Auto-Download**: Use `load_dataset_model_pair()` with `auto_download=True` to automatically download missing datasets
- **Path Configuration**: Download location is managed via `act.util.path_config.get_torchvision_data_root()`

For detailed documentation, see: `act/front_end/torchvision/README.md`
