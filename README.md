## Requirements

torch 2.0.1, mmcv, mmdet

CUDA_VISIBLE_DEVICES=7  python main.py --config consep.py --output_dir consep/del   --use_wandb  --run_name 1210_consep_base --pro_name CheckP   > 1210_consep_base.txt &


CUDA_VISIBLE_DEVICES=6 nohup python main.py --config pannuke123.py --output_dir pannuke123/0109_pannuke123_base    --use_wandb  --run_name 0109_pannuke123_base --pro_name pannuke   > 0109_pannuke123_base.txt &

CUDA_VISIBLE_DEVICES=6 nohup python main.py --config pannuke213.py --output_dir pannuke213/0109_pannuke213_base    --use_wandb  --run_name 0109_pannuke213_base --pro_name pannuke   > 0109_pannuke213_base.txt &

CUDA_VISIBLE_DEVICES=7 nohup python main.py --config pannuke321.py --output_dir pannuke321/0109_pannuke321_base    --use_wandb  --run_name 0109_pannuke321_base --pro_name pannuke   > 0109_pannuke321_base.txt &



CUDA_VISIBLE_DEVICES=6 python predict_prompts.py --config consep.py --resume /data/hotaru/projects/ORI_PromptNucSeg/prompter/checkpoint/1210_consep_base/best.pth

CUDA_VISIBLE_DEVICES=2 python main.py --config puma_nuclei10.py --output_dir puma_nuclei10/1208_puma10_1024  --use_wandb  --run_name 1208_puma10_1024  > 1208_puma10_1024.txt &




CUDA_VISIBLE_DEVICES=2 python predict_prompts.py --config puma_nuclei10.py --resume /data/hotaru/my_projects/ORI_PromptNucSeg/prompter/checkpoint/puma_nuclei10/1208_puma10_1024/best.pth



CUDA_VISIBLE_DEVICES=2 python predict_prompts.py --config pannuke123.py --resume /data/hotaru/projects/ORI_PromptNucSeg/prompter/checkpoint/dpa_pannuke123.pth
CUDA_VISIBLE_DEVICES=2 python predict_prompts.py --config pannuke213.py --resume /data/hotaru/projects/ORI_PromptNucSeg/prompter/checkpoint/dpa_pannuke213.pth
CUDA_VISIBLE_DEVICES=2 python predict_prompts.py --config pannuke321.py --resume /data/hotaru/projects/ORI_PromptNucSeg/prompter/checkpoint/dpa_pannuke321.pth

 
CUDA_VISIBLE_DEVICES=6 nohup  python main.py --config cpm17_b.py --output_dir cpm/0109_cpm_batch8_5ef5_lossc  --use_wandb --run_name 0109_cpm_batch8_5ef5_lossc  --pro_name PNS-M  > 0109_cpm_batch8_5ef5_lossc.txt &



CUDA_VISIBLE_DEVICES=5 nohup   python main.py --config pannuke321_h.py --output_dir pannuke/0110_pannuke321_seg_h  --use_wandb --run_name 0110_pannuke321_seg_h --pro_name pannuke  > 0110_pannuke321_seg_h.txt &

CUDA_VISIBLE_DEVICES=6  nohup  python main.py --config pannuke213_h.py --output_dir pannuke/0113_pannuke213_seg_h  --use_wandb --run_name 0113_pannuke213_seg_h --pro_name pannuke  > 0113_pannuke213_seg_h.txt &


CUDA_VISIBLE_DEVICES=3  nohup  python main.py --config consep_h.py --output_dir consep/0112_consep_seg_ori_h_512  --use_wandb --run_name 0112_consep_seg_ori_h_512  --pro_name CheckP  > 0112_consep_seg_ori_h_512.txt &

CUDA_VISIBLE_DEVICES=6 nohup  python main.py --config consep_b.py --output_dir consep/0104_consep_h  --use_wandb --pro_name CheckP --run_name 0104_consep_h > 0104_consep_h.txt &

CUDA_VISIBLE_DEVICES=2 nohup  python main.py --config puma_nuclei10_b.py --output_dir puma_nuclei10/1208_puma10_1024_seg  --use_wandb --run_name 1208_puma10_1024_seg > 1208_puma10_1024_seg.txt &

CUDA_VISIBLE_DEVICES=2  python main.py --config puma_nuclei10_b.py  --eval  --resume  /data/hotaru/my_projects/ORI_PromptNucSeg/segmentor/checkpoint/puma_nuclei10/1208_puma10_1024_seg/bestaji.pth



CUDA_VISIBLE_DEVICES=2  python main.py --config pannuke321_h.py  --eval  --resume  /data/hotaru/projects/ORI_PromptNucSeg/segmentor/checkpoint/pannuke/pannuke321_h.pth







## Dataset

1. download [Kumar](https://github.com/honglianghe/CDNet/issues/6), [CPM-17](https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA?usp=drive_link) and [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) datasets.

2. run **extract_data.py** for data pre-processing. 

   ```markdown
   datasets
   ├── cpm17
   │   ├── extract_data.py
   │   ├── test
   │   └── train
   ├── cpm17_test_files.npy
   ├── cpm17_train_files.npy
   ├── kumar
   │   ├── extract_data.py
   │   ├── images
   │   └── labels
   ├── kumar_test_files.npy
   ├── kumar_train_files.npy
   ├── pannuke
   │   ├── extract_data.py
   │   ├── Fold 1
   │   ├── Fold 2
   │   ├── Fold 3
   │   ├── Images
   │   └── Masks
   ├── pannuke123_test_files.npy
   ├── pannuke123_train_files.npy
   ├── pannuke123_val_files.npy
   ├── pannuke213_test_files.npy
   ├── pannuke213_train_files.npy
   ├── pannuke213_val_files.npy
   ├── pannuke321_test_files.npy
   ├── pannuke321_train_files.npy
   └── pannuke321_val_files.npy
   ```



## Training

1. Train the prompter

   ```shell
   cd prompter
   python main.py --config dpa_pannuke123.py --output_dir dpa_pannuke123
   # python main.py --config dpa_pannuke213.py --output_dir dpa_pannuke123
   # python main.py --config dpa_pannuke321.py --output_dir dpa_pannuke123
   ```

2. Use the trained prompter to generate nuclei prompts for the validation and test sets.

   ```shell
   python predict_prompts.py --config dpa_pannuke123.py --resume checkpoint/dpa_pannuke123/best.pth
   # python predict_prompts.py --config dpa_pannuke213.py --resume checkpoint/dpa_pannuke213/best.pth
   # python predict_prompts.py --config dpa_pannuke321.py --resume checkpoint/dpa_pannuke321/best.pth
   ```

3. Download SAM's pre-trained [weights](https://github.com/facebookresearch/segment-anything) into **segmentor/pretrained** and train the segmentor.

   ```shell
   cd segmentor
   torchrun --nproc_per_node=4 main.py --config pannuke123_b.py --output_dir pannuke123_b
   # torchrun --nproc_per_node=4 main.py --config pannuke213_b.py --output_dir pannuke213_b
   # torchrun --nproc_per_node=4 main.py --config pannuke321_b.py --output_dir pannuke321_b
   ```

   

## Evaluation

see [test.sh](https://github.com/windygoo/PromptNucSeg/blob/main/segmentor/test.sh)



## Checkpoints

|             |                            Kumar                             |                            CPM-17                            |                          PanNuke123                          |                          PanNuke213                          |                          PanNuke321                          |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Prompter   | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ee_5mPeYZIhGufpsumWbp1QBWPKLg6BxLoXoOzl9BGywVw?e=mHm8Wg) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ec0xXaiuz2JIjDInHq1tuEwBJKowhkaxUEqPUiQENeHmPA?e=DSynNh) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYtVl95nSypFvTJa8B5vSUIB9ibmgxwF9ACFNnDdjBWDXA?e=EvH5PS) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EbmGEmoL539HkBBPpC-SyagB4niZG9IlaNnF71mRuFqa7Q?e=9jHWY5) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EXskYXQgtFZOtu2t-FLzbC8BTqZr8QtRiqtcmOiWCZcpNg?e=05qYTo) |
| Segmentor-B | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EdDo45KYM9BPl3JGKsYaOZsB5lZOUzCZdy7jwZBxn4htGg?e=35cLRu) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EcgysPKrQP1Fs_oByRWAEngBuDjw3Kn6akZbtTl6Wj2hYg?e=nAb6za) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EfznT2AbW5VNiQIIfeq5h8sBvdoioMH35P9PA7bnF1igCQ) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EVvfF959JptOolf7xt8ZPdoBIGIM9UwpTCWDhSLVTtDV_w?e=bJuHZ5) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EQi4RBhTdvFGqIgNyj9UZ4QBTrmK9kJcLwJ4HMlPTHq53w?e=11r7IN) |
| Segmentor-L | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYLBmedg0nlAp5dqitn8pxcBo_9OcRWHOKpzb5Q9g5f8Kw?e=kie9IK) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/ET0D1YyinExLnum2L3y4soABiPgw_99AcocruqM4bw95pA?e=XVoDhq) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EWQ_o9jIIWVItezvJnpPmkEBQY38Agh0YGHlOHCQZGAIig?e=Foscbm) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EZ02oBK828dLo5P1Z1N9RV0BpzIum-8du7HXDCU4Ue8omg?e=Kt5v5r) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EWErh4qZWSxErgGGxZ_fQPQB3KXnGZ1iTJVtzwwn5sNJyg?e=8ZQo9m) |
| Segmentor-H | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EYCdndKjn4NJv3Qvebo4YsQBrUhU_Uu2tjtBucJH2SMdNQ?e=NThF4d) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/Ebg9v0HaOFZIpyda-JKNST8B2AmnGdhgYQqjdLHYm4j5LA?e=ibANRv) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EZjmiotww1hHtF83WwJTVz0BAmfDNkuSuGbUXkthP3yvDQ?e=N45aU3) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EQG3IMH1OARPj67mapQoakYBjlkMzAKzQjYxPn425JiVeQ?e=6XrKmT) | [OneDrive](https://westlakeu-my.sharepoint.com/:u:/g/personal/shuizhongyi_westlake_edu_cn/EVj-vCQh5MVPqIT8ggkSJGsBeWv_MsrO9Ci3Lr7wuewW2A?e=qsa0Gd) |

## Contact
If you have any questions or concerns, feel free to report issues or directly contact us (Zhongyi Shui shuizhongyi@westlake.edu.cn).

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@article{shui2023unleashing,
  title={Unleashing the Power of Prompt-driven Nucleus Instance Segmentation},
  author={Shui, Zhongyi and Zhang, Yunlong and Yao, Kai and Zhu, Chenglu and Sun, Yuxuan and Yang, Lin},
  journal={arXiv preprint arXiv:2311.15939},
  year={2023}
}
```

