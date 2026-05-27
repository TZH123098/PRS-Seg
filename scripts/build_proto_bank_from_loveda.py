import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
import open_clip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/LoveDA")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--class_file", type=str, default="./configs/cls_loveda.txt")
    parser.add_argument("--save_path", type=str, default="./weights/prototypes/loveda_k8.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_patches_per_class", type=int, default=500)
    parser.add_argument("--min_area", type=int, default=128)
    parser.add_argument("--min_crop_size", type=int, default=24)
    parser.add_argument("--openclip_model", type=str, default="ViT-B-16")
    parser.add_argument("--openclip_pretrained", type=str, default="./weights/openclip/open_clip_pytorch_model.bin")
    return parser.parse_args()


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def crop_masked_patch(image_np: np.ndarray, mask_np: np.ndarray, min_crop_size: int):
    bbox = mask_to_bbox(mask_np)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    if w < min_crop_size or h < min_crop_size:
        return None

    patch = image_np.copy()
    patch[mask_np == 0] = 0
    patch = patch[y1:y2 + 1, x1:x2 + 1]

    if patch.size == 0:
        return None

    return Image.fromarray(patch)


@torch.no_grad()
def encode_image(clip_model, preprocess, pil_img, device):
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    feat = clip_model.encode_image(img_tensor)
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0).cpu()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names = load_class_names(args.class_file)

    # 统一目录结构
    img_dir = os.path.join(args.data_root, "img_dir", args.split)
    ann_dir = os.path.join(args.data_root, "ann_dir", args.split)

    assert os.path.isdir(img_dir), f"img_dir not found: {img_dir}"
    assert os.path.isdir(ann_dir), f"ann_dir not found: {ann_dir}"

    # LoveDA 标签：0 是 ignore/no-data，1~7 才是有效类别
    valid_label_ids = list(range(1, len(class_names) + 1))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if os.path.isfile(args.openclip_pretrained):
        print(f"[ProtoBank] loading local OpenCLIP weights from: {args.openclip_pretrained}")
    else:
        print(f"[ProtoBank] loading OpenCLIP pretrained tag: {args.openclip_pretrained}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.openclip_model,
        pretrained=args.openclip_pretrained
    )
    clip_model = clip_model.to(device).eval()

    collected_feats = defaultdict(list)

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    print(f"[ProtoBank] found {len(image_files)} images in {img_dir}")

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(img_dir, img_name)

        stem = os.path.splitext(img_name)[0]
        ann_path = None
        for ext in [".png", ".tif", ".tiff", ".jpg"]:
            candidate = os.path.join(ann_dir, stem + ext)
            if os.path.exists(candidate):
                ann_path = candidate
                break

        if ann_path is None:
            print(f"[Warning] annotation not found for {img_name}, skip")
            continue

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        ann_np = np.array(Image.open(ann_path))
        if ann_np.ndim == 3:
            ann_np = ann_np[..., 0]

        for label_id in valid_label_ids:
            class_index = label_id - 1
            class_name = class_names[class_index]

            if len(collected_feats[class_name]) >= args.max_patches_per_class:
                continue

            mask = (ann_np == label_id).astype(np.uint8)
            area = int(mask.sum())

            if area < args.min_area:
                continue

            patch = crop_masked_patch(image_np, mask, args.min_crop_size)
            if patch is None:
                continue

            feat = encode_image(clip_model, preprocess, patch, device)
            collected_feats[class_name].append(feat)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            print(f"[ProtoBank] processed {idx + 1}/{len(image_files)}")

    proto_bank = {}

    for class_name in class_names:
        feats = collected_feats[class_name]

        if len(feats) == 0:
            print(f"[Warning] no patch collected for class: {class_name}")
            continue

        feats = torch.stack(feats, dim=0)
        feats_np = feats.numpy()

        k = min(args.k, len(feats_np))
        if k <= 1:
            centers = feats[:1]
        else:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(feats_np)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        centers = F.normalize(centers, dim=-1)
        proto_bank[class_name] = centers
        print(f"[ProtoBank] class={class_name}, num_feats={len(feats_np)}, num_proto={centers.shape[0]}")

    torch.save(proto_bank, args.save_path)
    print(f"[ProtoBank] saved to: {args.save_path}")


if __name__ == "__main__":
    main()