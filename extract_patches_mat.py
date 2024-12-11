import re
import glob
import os
import tqdm
import pathlib
import numpy as np
import scipy.io as sio  # 用于保存mat文件
import cv2  # 用于保存图像为PNG格式

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from prepare_get_dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    type_classification = True

    win_size = [256, 256]
    step_size = [204, 204] #164
    extract_type = "valid"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    dataset_name = "puma_nuclei10"
    save_root = "/data/hotaru/my_projects/ORI_PromptNucSeg/prompter/datasets/puma_nuclei10/train/%s/" % dataset_name

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".tif", "/data/hotaru/my_projects/ORI_PromptNucSeg/prompter/datasets/puma_nuclei10/train/images"),
            "ann": (".mat", "/data/hotaru/my_projects/ORI_PromptNucSeg/prompter/datasets/puma_nuclei10/train/labels"),
        },
        # "valid": {
        #     "img": (".tif", "/data/hotaru/projects/dataset/cpm17/test/Images"),
        #     "ann": (".mat", "/data/hotaru/projects/dataset/cpm17/test/Labels"),
        # },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)

    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_img_dir = "%s/%s/%s/Images/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        
        out_ann_dir = "%s/%s/%s/Labels/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()

        rm_n_mkdir(out_img_dir)
        rm_n_mkdir(out_ann_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(total=len(file_list), bar_format=pbar_format, ascii=True, position=0)

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            # ann = parser.load_ann("%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification)
            ann = parser.load_ann("%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification)

            sub_patches_img = xtractor.extract(img, extract_type)
            sub_patches_ann = xtractor.extract(ann, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(total=len(sub_patches_img), leave=False, bar_format=pbar_format, ascii=True, position=1)

            for idx, (patch_img, patch_ann) in enumerate(zip(sub_patches_img, sub_patches_ann)):
                img_save_path = "{0}/{1}_{2:03d}.tif".format(out_img_dir, base_name, idx)
                ann_save_path = "{0}/{1}_{2:03d}.mat".format(out_ann_dir, base_name, idx)

                # 使用OpenCV保存图像为tif
                # patch_img = (patch_img * 255).astype(np.uint8)  # 确保图像格式正确
                patch_img_bgr = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)  # 转换为BGR
                cv2.imwrite(img_save_path, patch_img_bgr)

                # 保存label为mat文件
                if type_classification:
                    inst_map = patch_ann[..., 0]
                    type_map = patch_ann[..., 1]
                    # print(np.unique(type_map))
                    inst_map = np.squeeze(inst_map) if inst_map.shape[-1] == 1 else inst_map
                    type_map = np.squeeze(type_map) if type_map.shape[-1] == 1 else type_map
                    print(inst_map.shape)

                    # Save the instance map and type map into .mat file
                    sio.savemat(ann_save_path, {'inst_map': inst_map, 'type_map': type_map})
                else:
                    # inst_map = np.squeeze(inst_map) if inst_map.shape[-1] == 1 else inst_map
                    inst_map = patch_ann[..., 0]
                    inst_map = np.squeeze(inst_map) if inst_map.shape[-1] == 1 else inst_map
                    sio.savemat(ann_save_path, {'inst_map':inst_map})
                    print(patch_ann[..., 0].shape)
                    print(inst_map.shape)

                pbar.update()

            pbar.close()
            pbarx.update()
            # break

        pbarx.close()
