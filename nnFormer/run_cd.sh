python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 0
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 1
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 2
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 0 -val
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 1 -val
python ./nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_PelvicTumour 11 2 -val


## CoTr
python ./CoTr/run/run_training.py 3d_fullres nnUNetTrainerV2_ResTrans 11 0