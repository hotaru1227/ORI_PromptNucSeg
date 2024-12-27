
import torch
import PIL.Image
import sys
sys.path.append('/data/hotaru/projects/ORI_PromptNucSeg/CLIP') 
import clip
import os
import cv2
from typing import List
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# 移除缓存，避免影响后续调用
# @lru_cache

def load_clip(name: str = "ViT-B/16"):   # RN50x64   ViT-L/14@336px   ViT-B/32
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess

@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str], model, preprocess) -> torch.Tensor:
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    # print(preprocessed.shape)
    tokens = clip.tokenize(texts).to(device)
    # logits_per_image, _ = model(preprocessed, tokens)

    with torch.no_grad():
        image_features = model.encode_image(preprocessed)
        text_features = model.encode_text(tokens)
    

    x_tokens = image_features[:, 1:, :]
    h, w = 14, 14  # 假设是 14x14 patches
    x_2d = x_tokens.view(1, h, w, 512)  # 变换为 [1, 14, 14, 512]
    print("reshaped", x_2d.shape)
    patch_features_2d = x_2d.permute(0, 3, 1, 2)  # 转为 [1, 512, 14, 14]
    print("patch_features_2d", patch_features_2d.shape)
    import torch.nn.functional as F
    pixel_features = F.interpolate(patch_features_2d, size=(224, 224), mode="bilinear", align_corners=False)
    print("pixel_features", pixel_features.shape)
    print("text_features",text_features.shape)


    output_dir = "/data/hotaru/projects/ORI_PromptNucSeg/tmp_nuclear"
    os.makedirs(output_dir, exist_ok=True)

    aggregated_map = pixel_features[0].mean(dim=0).detach().cpu().numpy()  # 聚合所有通道
    print("aggregated_map", aggregated_map.shape)
    aggregated_map = aggregated_map.astype(np.float32)

    '''可视化聚合的image特征aggregated_map'''
    # 归一化 aggregated_map 到 [0, 1] 范围
    normalized_aggregated = Normalize(vmin=aggregated_map.min(), vmax=aggregated_map.max())(aggregated_map)

    # 使用 matplotlib 的 jet 颜色映射
    cmap = plt.get_cmap("jet")
    colored_aggregated = cmap(normalized_aggregated)  # Shape: (H, W, 4), 包括 RGBA

    # 保存彩色图像
    output_path = os.path.join(output_dir, "aggregated_feature_map_colored.png")
    plt.imsave(output_path, colored_aggregated[..., :3])  # 仅保存 RGB 通道

    # 可视化并保存
    plt.figure(figsize=(8, 8))
    plt.imshow(colored_aggregated, interpolation="nearest")
    plt.axis("off")
    plt.title("Aggregated Feature Map (Colored)")
    plt.savefig(os.path.join(output_dir, "aggregated_feature_map_visualization.png"), bbox_inches="tight")
    print(f"Visualizations saved in {output_dir}")
    '''1'''


    h, w = patch_features_2d.shape[2:] 
    pixel_features = patch_features_2d.permute(0, 2, 3, 1).reshape(-1, 512)  # [224*224, 512]

    pixel_features_norm = F.normalize(pixel_features, p=2, dim=1)  # [224*224, 512]
    text_features_norm = F.normalize(text_features, p=2, dim=1)  # [1, 512]

    S = torch.mm(pixel_features_norm, text_features_norm.T)  # [224*224, 1]
    
    S_reshaped = S.view(h, w)  # [224, 224]

    # 对 S_reshaped 进行 min-max normalization
    P = (S_reshaped - S_reshaped.min()) / (S_reshaped.max() - S_reshaped.min())  # [224, 224]

    P_numpy = P.cpu().numpy()
    P_tensor = torch.from_numpy(P_numpy).float().unsqueeze(0).unsqueeze(0)
    P_numpy = F.interpolate(P_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    '''可视化P_numpy'''
    P_numpy = P_numpy.squeeze().detach().cpu().numpy()  # 去掉 batch 维度并转换为 NumPy 格式

    # 标准化到 [0, 1] 范围
    P_normalized = (P_numpy - P_numpy.min()) / (P_numpy.max() - P_numpy.min())
    output_path = os.path.join(output_dir, "p.png")
    plt.imsave(output_path, P_normalized, cmap="jet")  # 仅保存 RGB 通道
    # 可视化并保存
    plt.figure(figsize=(8, 8))
    plt.imshow(P_normalized, cmap="jet", interpolation="nearest")
    plt.axis("off")
    plt.title("Visualization of P_numpy")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "P_numpy_visualization.png"), bbox_inches="tight", pad_inches=0)

    print(f"Visualization saved at {os.path.join(output_dir, 'P_numpy_visualization.png')}")
    '''1'''

# 主程序
if __name__ == "__main__":
    model, preprocess = load_clip()

    image_path = "/data/hotaru/projects/ORI_PromptNucSeg/segmentor/datasets/pannuke/Images/1_0.png"

    texts = "nuclear"
    # texts = "Nuclei with irregular, polygonal, or elliptical shapes and uneven chromatin distribution, exhibiting staining regions with varying intensities in a patchy pattern."
    save_path = "/data/hotaru/projects/ORI_PromptNucSeg/tmp/R.jpg"

    # 打开图像
    image = PIL.Image.open(image_path).convert("RGB")

    
    score = get_score(image, texts, model, preprocess)
    # print(f"Global similarity score: {score}")\


    '''huggingface的clip'''
    # patch feature：torch.Size([1, 196, 768])
    # text feature： torch.Size([1, 8, 512])

    # from transformers import CLIPProcessor, CLIPVisionModel
    # from transformers import CLIPTextModel, CLIPTokenizer

    # # 加载 CLIP 的视觉模型
    # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # model_name = "openai/clip-vit-base-patch16"
    # tokenizer = CLIPTokenizer.from_pretrained(model_name)
    # text_model = CLIPTextModel.from_pretrained(model_name)
    # # 输入图像预处理
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)

    # # 提取 patch 特征
    # patch_features = outputs.last_hidden_state[:, 1:, :]  # 跳过 CLS token
    # print("222,",patch_features.shape)
    # num_patches = int(patch_features.shape[1] ** 0.5)  # Patch 格子数
    # patch_features_2d = patch_features.view(-1, num_patches, num_patches, 768)

    # print("333",patch_features_2d.shape)
    # patch_features_2d = patch_features_2d.permute(0, 3, 1, 2)  # 转为 [1, 768, 14, 14]
    # import torch.nn.functional as F
    # pixel_features = F.interpolate(patch_features_2d, size=(224, 224), mode="bilinear", align_corners=False)
    # print(pixel_features.shape)  # [1, 768, 224, 224]


    # texts = ["Cells in pathological tissue."]

    # # Tokenize input
    # inputs = tokenizer(texts, return_tensors="pt", padding=True)

    # # Get text features
    # text_features = text_model(**inputs).last_hidden_state
    # print("Text Features Shape:", text_features.shape)

    # import os
    # import cv2
    # # 创建保存目录
    # output_dir = "/data/hotaru/projects/ORI_PromptNucSeg/tmp"
    # os.makedirs(output_dir, exist_ok=True)

    # # # 选择一个通道进行可视化（如第一个通道）
    # # channel_idx = 0
    # # feature_map = pixel_features[0, channel_idx, :, :].detach().cpu().numpy()  # 转为 NumPy 格式
    # # # 归一化到 [0, 255]
    # # normalized_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # # # 保存为灰度图
    # # cv2.imwrite(os.path.join(output_dir, f"feature_channel_{channel_idx}.png"), normalized_map)
    # # # 可视化伪彩色图
    # # colored_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
    # # cv2.imwrite(os.path.join(output_dir, f"feature_channel_{channel_idx}_colored.png"), colored_map)

    # # 或者将所有通道平均聚合后保存
    # aggregated_map = pixel_features[0].mean(dim=0).detach().cpu().numpy()  # 聚合所有通道
    # normalized_aggregated = cv2.normalize(aggregated_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite(os.path.join(output_dir, "aggregated_feature_map.png"), normalized_aggregated)

    # colored_aggregated = cv2.applyColorMap(normalized_aggregated, cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join(output_dir, "aggregated_feature_map_colored.png"), colored_aggregated)

    # print(f"Visualizations saved in {output_dir}")




