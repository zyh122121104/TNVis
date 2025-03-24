# TNVis
Thyroid nodules visualization.
Source code of the paper ["A Deep-learning Based Ultrasound Diagnostic Tool for Three-Dimensional Visualization of Thyroid Nodules: A Multicenter Diagnostic Study"](https://www.nature.com/articles/s41746-025-01455-y).
# Download pre-trained swin transformer model (Swin-T)
[Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"
# Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
# Train/Test
Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

Train

`python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 300 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24`

Test

`python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 300 --base_lr 0.05 --img_size 224 --batch_size 24`

# Citations
If you find this work helpful, please cite the following paper:

**Zhou, Yahan**, Chen, Chen, Yao, Jincao, et al. (2025). *A deep learning based ultrasound diagnostic tool driven by 3D visualization of thyroid nodules*. *npj Digital Medicine*, 8(1), 126.  
[DOI: 10.1038/s41746-025-01455-y](https://doi.org/10.1038/s41746-025-01455-y)
