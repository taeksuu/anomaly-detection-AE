# anomaly-detection

## Results
* Image-level anomaly detection accuracy (ROCAUC)

|Category|Original_L2_128|Original_SSIM_128|Large_L2_2048|Large_SSIM_2048|
|:---:|:---:|:---:|:---:|:---:|
|Carpet| | | | |
|Grid| | | | |
|Leather| | | | |
|Tile| | | | |
|Wood| | | | |
|All texture classes| | | | |
|Bottle| | | | |
|Cable| | | | |
|Capsule| | | | |
|Hazelnut| | | | |
|Metal nut| | | | |
|Pill| | | | |
|Screw| | | | |
|Toothbrush| | | | |
|Transistor| | | | |
|Zipper| | | | |
|All object classes| | | | |
|All classes| | | | |

* Pixel-level anomaly detection accuracy (ROCAUC)

|Category|Original_L2_128|Original_SSIM_128|Large_L2_2048|Large_SSIM_2048|
|:---:|:---:|:---:|:---:|:---:|
|Carpet| | | | |
|Grid| | | | |
|Leather| | | | |
|Tile| | | | |
|Wood| | | | |
|All texture classes| | | | |
|Bottle| | | | |
|Cable| | | | |
|Capsule| | | | |
|Hazelnut| | | | |
|Metal nut| | | | |
|Pill| | | | |
|Screw| | | | |
|Toothbrush| | | | |
|Transistor| | | | |
|Zipper| | | | |
|All object classes| | | | |
|All classes| | | | |

* Per-region-overlap detection accuracy (PROAUC)

|Category|Original_L2_128|Original_SSIM_128|Large_L2_2048|Large_SSIM_2048|
|:---:|:---:|:---:|:---:|:---:|
|Carpet| | | | |
|Grid| | | | |
|Leather| | | | |
|Tile| | | | |
|Wood| | | | |
|All texture classes| | | | |
|Bottle| | | | |
|Cable| | | | |
|Capsule| | | | |
|Hazelnut| | | | |
|Metal nut| | | | |
|Pill| | | | |
|Screw| | | | |
|Toothbrush| | | | |
|Transistor| | | | |
|Zipper| | | | |
|All object classes| | | | |
|All classes| | | | |

* Precision-recall detection accuracy (PRAUC)

|Category|Original_L2_128|Original_SSIM_128|Large_L2_2048|Large_SSIM_2048|
|:---:|:---:|:---:|:---:|:---:|
|Carpet| | | | |
|Grid| | | | |
|Leather| | | | |
|Tile| | | | |
|Wood| | | | |
|All texture classes| | | | |
|Bottle| | | | |
|Cable| | | | |
|Capsule| | | | |
|Hazelnut| | | | |
|Metal nut| | | | |
|Pill| | | | |
|Screw| | | | |
|Toothbrush| | | | |
|Transistor| | | | |
|Zipper| | | | |
|All object classes| | | | |
|All classes| | | | |


### Metrics Curve

* Large_L2_2048 - SSIM

<p align="left">
    <img src="imgs/metrics_curve.png" width="1000"\>
</p>


### Segmentation examples
* Large_L2_2048 - L1
<p align="left">
    <img src="imgs/hazelnut_0_l1.png" width="600"\>
</p>
* Large_L2_2048 - L2
<p align="left">
    <img src="imgs/hazelnut_0_l2.png" width="600"\>
</p>
* Large_L2_2048 - SSIM
<p align="left">
    <img src="imgs/hazelnut_0_ssim.png" width="600"\>
</p>
* Large_L2_2048 - L1 + L2 + SSIM
<p align="left">
    <img src="imgs/hazelnut_0_all.png" width="600"\>
