# ACMTR：Attention-guided combined multi-scale transformer-based network for 3D CT pelvic functional bone marrow segmentation
Code for paper “ACMTR: Attention-guided, combined multi-scale, transformer reasoning-based network for 3D CT pelvic functional bone marrow segmentation” Please read our published version at the following link: https://doi.org/10.1016/j.bspc.2022.104522

Parts of codes are borrowed from [nnFormer](https://github.com/282857341/nnFormer), about dataset you can follow the settings in nnUNet for path configurations and preprocessing procedures [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md).

## Training
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 0

## Val
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 0 -val
