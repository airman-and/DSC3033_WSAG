import os
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
from PIL import Image
from models.locate import Net as model

from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/root/workspace/andycho/CV/AGD20K')
parser.add_argument('--model_file', type=str, default='path_to_ckpt')
parser.add_argument('--save_path', type=str, default='./save_preds')
parser.add_argument('--save_visuals', action='store_true')
parser.add_argument('--save_heatmaps', action='store_true')
parser.add_argument('--max_save_images', type=int, default=None)
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
#### test
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--gamma1', type=float, default=0.6)
parser.add_argument('--gamma2', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--cont_temperature', type=float, default=0.5)


args = parser.parse_args()

if args.divide == "Seen":
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
else:
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]

if args.divide == "Seen":
    args.num_classes = 36
else:
    args.num_classes = 25

args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

def post_process(KLs, SIM, NSS, ego_pred, GT_mask, args):
    ego_pred = np.array(ego_pred.squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
    KLs.append(kld)
    SIM.append(sim)
    NSS.append(nss)
    return KLs, SIM, NSS, kld


def pred_to_map(pred, crop_size):
    pred = np.array(pred.squeeze().data.cpu())
    return normalize_map(pred, crop_size)


def map_to_heatmap(pred_map, image_size):
    width, height = image_size
    pred_map = cv2.resize(pred_map, (width, height), interpolation=cv2.INTER_LINEAR)
    pred_uint8 = np.clip(pred_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def image_path_from_mask_path(mask_path, args):
    rel_path = os.path.relpath(mask_path, args.mask_root)
    rel_stem = os.path.splitext(rel_path)[0]
    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = os.path.join(args.test_root, rel_stem + ext)
        if os.path.exists(image_path):
            return image_path
    raise FileNotFoundError("Could not find image for mask: {}".format(mask_path))


def prepare_visual_dirs(args):
    overlay_dir = os.path.join(args.save_path, "overlay")
    heatmap_dir = os.path.join(args.save_path, "heatmap")
    os.makedirs(overlay_dir, exist_ok=True)
    if args.save_heatmaps:
        os.makedirs(heatmap_dir, exist_ok=True)
    return overlay_dir, heatmap_dir


def save_visualizations(predictions, mask_path, step, args, overlay_dir, heatmap_dir):
    if args.max_save_images is not None and step >= args.max_save_images:
        return

    image_path = image_path_from_mask_path(mask_path, args)
    image = Image.open(image_path).convert("RGB")
    rel_path = os.path.relpath(mask_path, args.mask_root)
    stem = os.path.splitext(rel_path)[0].replace(os.sep, "_")
    image_np = np.asarray(image).astype(np.float32)

    for pred_name, pred in predictions:
        pred_map = pred_to_map(pred, args.crop_size)
        heatmap = map_to_heatmap(pred_map, image.size)
        overlay = (0.55 * image_np + 0.45 * heatmap.astype(np.float32)).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, "{}_{}.png".format(stem, pred_name)))

        if args.save_heatmaps:
            Image.fromarray(heatmap).save(os.path.join(heatmap_dir, "{}_{}.png".format(stem, pred_name)))


if __name__ == '__main__':
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes, args=args).cuda()

    KLs = []
    SIM = []
    NSS = []
    reeKLS = []
    remKLS = []
    reeSIM = []
    remSIM = []
    reeNSS = []
    remNSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file), strict=False)

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")
    overlay_dir, heatmap_dir = None, None
    if args.save_visuals:
        overlay_dir, heatmap_dir = prepare_visual_dirs(args)

    for step, (image, label, mask_path) in enumerate(tqdm(TestLoader)):
        names = mask_path[0].split("/")
        key = names[-3] + "_" + names[-2] + "_" + names[-1]
        GT_mask = GT_masks[key]
        GT_mask = GT_mask / 255.0

        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))
        with torch.no_grad():
            ego_pred, refined_CLIP_ego_ego, refined_CLIP_ego_mean = model.test_forward(image.cuda(), label.long().cuda())

        if args.save_visuals:
            save_visualizations([
                ("ego_pred", ego_pred),
                ("refined_ego", refined_CLIP_ego_ego),
                ("refined_mean", refined_CLIP_ego_mean),
            ], mask_path[0], step, args, overlay_dir, heatmap_dir)

        KLs, SIM, NSS, _ = post_process(KLs, SIM, NSS, ego_pred, GT_mask, args)
        reeKLS, reeSIM, reeNSS, _ = post_process(reeKLS, reeSIM, reeNSS, refined_CLIP_ego_ego, GT_mask, args)
        remKLS, remSIM, remNSS, _ = post_process(remKLS, remSIM, remNSS, refined_CLIP_ego_mean, GT_mask, args)

    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    mreeKLS = sum(reeKLS) / len(reeKLS)
    mreeSIM = sum(reeSIM) / len(reeSIM)
    mreeNSS = sum(reeNSS) / len(reeNSS)

    mremKLS = sum(remKLS) / len(remKLS)
    mremSIM = sum(remSIM) / len(remSIM)
    mremNSS = sum(remNSS) / len(remNSS)

    print(f"KLD, SIM, NSS, {round(mKLD, 3)}, {round(mSIM, 3)}, {round(mNSS, 3)}")
    print(f"reeKLD, reeSIM, reeNSS, {round(mreeKLS, 3)}, {round(mreeSIM, 3)}, {round(mreeNSS, 3)}")
    print(f"remKLD, remSIM, remNSS, {round(mremKLS, 3)}, {round(mremSIM, 3)}, {round(mremNSS, 3)}")
