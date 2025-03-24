# TNVis
Thyroid nodules visualization.
Source code of the paper "A Deep-learning Based Ultrasound Diagnostic Tool for Three-Dimensional Visualization of Thyroid Nodules: A Multicenter Diagnostic Study".
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
`@article{zhou2025deep, 

  title={A deep learning based ultrasound diagnostic tool driven by 3D visualization of thyroid nodules}, 
  
  author={Zhou, Yahan and Chen, Chen and Yao, Jincao and Yu, Jiabin and Feng, Bojian and Sui, Lin and Yan, Yuqi and Chen, Xiayi and Liu, Yuanzhen and Zhang, Xiao and others}, 
  
  journal={npj Digital Medicine}, 
  
  volume={8}, 
  
  number={1}, 
  
  pages={126}, 
  
  year={2025} 
  
}`
