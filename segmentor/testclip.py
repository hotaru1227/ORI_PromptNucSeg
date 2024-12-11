
import torch
import PIL.Image
import clip
from typing import List
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# 移除缓存，避免影响后续调用
# @lru_cache

def load_clip(name: str = "ViT-L/14@336px"):
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess

@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str], model, preprocess) -> torch.Tensor:
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    print(preprocessed.shape)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)

    with torch.no_grad():
        image_features = model.encode_image(preprocessed)
        text_features = model.encode_text(tokens)
    
    print(image_features.shape)
    print(text_features.shape)
    


    similarity_map = 1

    # 生成热图来表示对齐结果
    # plt.imshow(similarity_map, cmap='hot', aspect='auto')
    # plt.colorbar(label="Similarity")
    # plt.title("Text-Image Alignment")
    # plt.savefig("/data/hotaru/my_projects/ORI_PromptNucSeg/segmentor/tmp/similarity_features.png", bbox_inches="tight")
    # plt.close()


    
    # 查看 logits_per_image 的值，确保它们有合理的范围
    print(f"logits_per_image: {logits_per_image}")
    
    similarity = logits_per_image.softmax(-1).cpu()
    print(f"Similarity (after softmax): {similarity}")
    return similarity[0, 0].item()

@torch.no_grad()
def get_score1(
    image: PIL.Image.Image, 
    text: str, 
    model, 
    preprocess, 
    crop_size: int, 
    save_path: str
):
    """
    对输入图像进行裁剪，计算所有裁剪区域与文本的相似度得分，并保存热图。

    Args:
        image (PIL.Image.Image): 输入图像。
        text (str): 待比较的文本。
        model: CLIP 模型。
        preprocess: CLIP 的预处理函数。
        crop_size (int): 裁剪区域的大小。
        save_path (str): 热图保存路径。
    """
    img_width, img_height = image.size

    # 保存所有裁剪的图像
    crops = []
    crop_coords = []  # 记录每个裁剪的 (x, y) 坐标
    for y in range(0, img_height, crop_size):
        for x in range(0, img_width, crop_size):
            crop = image.crop((x, y, min(x + crop_size, img_width), min(y + crop_size, img_height)))
            crops.append(preprocess(crop).unsqueeze(0))
            crop_coords.append((x, y))

    # 拼接所有裁剪的图像到一个 tensor
    input_tensor = torch.cat(crops, dim=0).to(device)
    tokens = clip.tokenize([text] * input_tensor.size(0)).to(device)

    # 使用模型预测 logits
    logits_per_image, _ = model(input_tensor, tokens)

    # 计算 softmax 归一化得分
    similarities = logits_per_image.softmax(dim=0).cpu().numpy()

    # 构建空的二维网格
    grid_height = (img_height + crop_size - 1) // crop_size  # 向上取整
    grid_width = (img_width + crop_size - 1) // crop_size    # 向上取整
    scores = np.zeros((grid_height, grid_width))

    # 将分数填充到网格中
    for i, (x, y) in enumerate(crop_coords):
        grid_y = y // crop_size
        grid_x = x // crop_size
        scores[grid_y, grid_x] = similarities[i, 0]

    # 可视化相似度热图
    plt.figure(figsize=(10, 8))
    plt.imshow(scores, cmap="viridis", extent=(0, img_width, img_height, 0))
    plt.colorbar(label="Similarity Score")
    plt.title(f"Heatmap of Text-Image Similarity ({text})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# 主程序
if __name__ == "__main__":
    model, preprocess = load_clip()

    image_path = "/data/hotaru/my_projects/ORI_PromptNucSeg/segmentor/datasets/cpm17_256/test/Images/image_00.png"

    texts = ["nuclei","dog"]
    save_path = "/data/hotaru/my_projects/ORI_PromptNucSeg/segmentor/tmp/similarity_heatmap_4.png"

    # 打开图像
    image = PIL.Image.open(image_path).convert("RGB")

    # # 计算裁剪相似度并保存热图
    # crop_size = 4 # 较小的裁剪尺寸
    # get_score1(image, text, model, preprocess, crop_size, save_path)
    # print(f"Heatmap saved to {save_path}")


    # texts = ["dogs","nuclei"]
    # image = PIL.Image.open(image_path).convert("RGB")
    
    score = get_score(image, texts, model, preprocess)
    # print(f"Global similarity score: {score}")
