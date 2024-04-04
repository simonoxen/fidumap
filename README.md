*work in progress*

# FiduMap

This project is largely inspired by [KeyMorph](https://github.com/alanqrwang/keymorph).

## Run

Trained models are provided in the Releases page, and are downloaded when running `fidumap.eval.register`.

```
python fidumap/eval/register.py --n_keypoints 32 --model default --moving moving.nii --fixed fixed.nii --out_prefix ~/out
```

## Train

Currently, this model is trained with *afids* and *IXI* datasets. To train locally, these data should be downloaded.

```
make train
```
