import os.path

import torch
import numpy as np
from tqdm import tqdm
from skimage import io
from utils import predict, mkdir
from models.dpa_p2pnet import build_model

from main import parse_args
from mmengine.config import Config


def crop_with_overlap(
        img,
        split_width,
        split_height,
        overlap
):
    split_width = 256
    split_height = 256
    def start_points(
            size,
            split_size,
            overlap
    ):
        points = [0]
        counter = 1
        stride = 256 - overlap
        split_size = 256
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    # print(img.shape)
    _,_,img_h, img_w = img.shape

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    crop_boxes = []
    for y in Y_points:
        for x in X_points:
            crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
    return np.asarray(crop_boxes)


args = parse_args()
cfg = Config.fromfile(f'config/{args.config}')

dataset = cfg.data.name
device = torch.device(args.device)

model = build_model(cfg)
# ckpt = torch.load(f'checkpoint/{args.resume}/best.pth', map_location='cpu')
ckpt = torch.load(f'{args.resume}', map_location='cpu')
pretrained_state_dict = ckpt['model']

model.load_state_dict(pretrained_state_dict)
model.eval()
model.to(device)

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=cfg.prompter.space,
                  pad_width_divisor=cfg.prompter.space, position="top_left", p=1),
    A.Normalize(),
    ToTensorV2()
], p=1)


def process_files(files):
    for file in tqdm(files):
        # img = io.imread(f'../segmentor/{file}')[..., :3]

        # image = transform(image=img)['image'].unsqueeze(0).to(device)
        # 发现npy文件保存的有问题，图像的拓展名写成了mat
        # 分割文件名和扩展名
        filename, extension = os.path.splitext(file)
        
        # 定义新的扩展名
        new_extension = '.png'  # 修改为你想要的新扩展名
        
        # 构造新的文件名
        new_file = filename + new_extension
        img = io.imread(f'{new_file}')[..., :3]

        image = transform(image=img)['image'].unsqueeze(0).to(device)

        crop_boxes = crop_with_overlap(
            image,
            (256,256),(256,256),
            int(64),
        ).tolist()

        #单张大图 point的结果列表
        all_points = []
        all_points_scores = []
        all_points_class = []
 
        processed_boxes = []
        

        for idx, crop_box in enumerate(crop_boxes):
            x1, y1, x2, y2 = crop_box
            crop_box_tuple = tuple(crop_box)
            pd_points, pd_scores, pd_classes, pd_masks = predict(
                model,
                image[..., y1:y2, x1:x2].to(device),
                ori_shape=np.array((y2 - y1, x2 - x1)),
                filtering=cfg.test.filtering,
                nms_thr=cfg.test.nms_thr,
            )

            pd_points[:, 0] += x1
            pd_points[:, 1] += y1

            # 检查pd_points是否出现在之前的任何一个box里
            bool_mask = np.ones(len(pd_points), dtype=bool)
            for prev_box in processed_boxes:
                px1, py1, px2, py2 = prev_box
                # 如果pd_points在之前的box里，设置mask为False
                bool_mask &= ~((pd_points[:, 0] >= px1+1) & (pd_points[:, 0] <= px2-1) &
                        (pd_points[:, 1] >= py1+1) & (pd_points[:, 1] <= py2-1))
                

            pd_points = pd_points[bool_mask]
            pd_scores = pd_scores[bool_mask]
            pd_classes = pd_classes[bool_mask]

            all_points.append(pd_points)
            all_points_scores.append(pd_scores)
            all_points_class.append(pd_classes)
            processed_boxes.append(crop_box)

        all_points = np.vstack(all_points)
        all_points_scores = np.concatenate(all_points_scores)
        all_points_class = np.concatenate(all_points_class)
        pd_points = all_points
        prompt_points = torch.from_numpy(pd_points).unsqueeze(1)
        pd_scores = all_points_scores
        pd_classes = all_points_class

        save_content = np.concatenate([pd_points, pd_classes[:, None]], axis=-1)

        np.save(
            f'../segmentor/prompts/{dataset}/{file.split("/")[-1][:-4]}',
            save_content
        )


mkdir(f'../segmentor/prompts/{dataset}')


test_files = np.load(f'../segmentor/datasets/{dataset}_test_files.npy')
process_files(test_files)

try:
    val_files = np.load(f'../segmentor/datasets/{dataset}_val_files.npy')
    process_files(val_files)

except FileNotFoundError:
    pass


