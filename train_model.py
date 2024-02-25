import argparse
import time

import dataset_location
import losses
import torch
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
from utils import render_from_angle


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--start_shape", default="sphere", type=str)
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")
        
    test_vis = []
    test_vis_rend = []
    for i in test_vis: test_vis_rend.append([])
    vis_step = 100
    if args.visualize:
        test_loader = torch.utils.data.DataLoader(
            r2n2_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_batched_R2N2,
            pin_memory=True,
            drop_last=True)
        eval_loader = iter(test_loader)
        max_iter = len(eval_loader)
        for step in range(max_iter):
            feed_dict = next(eval_loader)
            if step % vis_step == 0:
                images = feed_dict['images'].to(args.device).squeeze(1)
                images_rgb = images
                if args.load_feat:
                    images = torch.stack(feed_dict['feats']).to(args.device)
                test_vis.append((images, images_rgb))
    
    delta_theta = 360 * args.max_iter / args.save_freq
    theta = 0
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0: # and step > 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{args.type}_{args.start_shape}.pth",
            )
            if args.visualize:
                for i in range(len(test_vis)):
                    test_img = test_vis[i]
                    prediction = model(test_img[0], args)[0]
                    rend = render_from_angle(prediction, theta)
                    test_vis_rend[i].append(rend)
                theta += delta_theta
                    

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )
    
    if args.visualize:
        curr = 0
        for ani in test_vis_rend:
            imageio.mimsave(f"training_vis_{curr}.gif", ani, fps = 3, loop = 0)
            curr += vis_step
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
