# TNVis
Thyroid nodules visualization.
Source code of the paper "A Deep-learning Based Ultrasound Diagnostic Tool for Three-Dimensional Visualization of Thyroid Nodules: A Multicenter Diagnostic Study".
# Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
# Train/Test
Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

Train
`sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24`
Test
`sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24`
