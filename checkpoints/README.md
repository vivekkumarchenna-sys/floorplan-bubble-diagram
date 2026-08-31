# Checkpoints

Download the pretrained model weights and place them here:

```
checkpoints/
  segformer/
    best_model.pth       (541 MB, SegFormer-B3, val mIoU = 0.965)
  deeplab/
    best_deeplab.pth     (678 MB, DeepLabV3+ ResNet-101, val mIoU = 0.968)
```

**Download:** [Zenodo](https://doi.org/10.5281/zenodo.21600699)

These checkpoints are trained on the ResPlan dataset (17,107 floor plans, 16 classes, 512x512).

## SegFormer-B0 (lighter-backbone baseline)

`checkpoints/segformer_b0/best_model.pth` in the Zenodo archive. 3,718,256 trainable
parameters, best validation mIoU 0.9602 at epoch 18, trained under the identical protocol
(paper Appendix B.4, Table B.3). Reproduce its test-split numbers with:

```bash
python src/eval_backbone.py --ckpt checkpoints/segformer_b0/best_model.pth                             --pretrained nvidia/segformer-b0-finetuned-ade-512-512
```
