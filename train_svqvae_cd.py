"""
Natural training process including class divergence loss: 
margin-based approach with ramp
"""

import argparse
import os
import csv
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import distributed as dist
from models.s_vqvae_emb256 import S_VQVAE    ### adjust for respective model
from scheduler import CycleScheduler
from sklearn.model_selection import train_test_split

# class divergence loss function with margins
def class_divergence_loss(
    embed_code,
    labels,
    num_classes,
    device,
    margin_intra=0.5,   # max margin allowed intra-class distance
    margin_inter=1.5,   # min margin required inter-class distance
    epsilon=1e-5
):
    """
    Margin-based class divergence:
      - Intra-class margin (margin_intra): 
          If class's avg distance from centroid > margin_intra: penalize the excess
      - Inter-class margin (margin_inter):
          If distance between two class centroids < margin_inter: penalize the shortfall

    Args:
        embed_code: Tensor of shape (B, C, H, W). 
        labels: Long tensor of shape (B,) with class indices
        num_classes: number of classes (10 for CIFAR-10)
        device: 'cpu' or 'cuda'
        margin_intra: float, maximum intra-class distance
        margin_inter: float, minimum inter-class distance

    Returns:
        A scalar margin-based penalty that is 0 if all classes are 
        within margin_intra of their centroid, and all centroids are at 
        least margin_inter apart. Otherwise grows with the violation amount.
    """
    B, C, H, W = embed_code.shape
    # flatten the spatial dims and average to get a single vector per sample
    embed_code_flat = embed_code.view(B, C, -1).mean(dim=2)  # (B, C)

    # compute each class centroid
    class_centroids = torch.zeros(num_classes, C, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    for i in range(num_classes):
        mask = (labels == i)
        if mask.sum() > 0:
            class_centroids[i] = embed_code_flat[mask].mean(dim=0)
            class_counts[i] = mask.sum()

    valid_classes = (class_counts > 0)

    # intra-class penalty
    intra_class_loss = 0.0
    # inter-class penalty
    inter_class_loss = 0.0

    # intra class distances
    for i in range(num_classes):
        mask_i = (labels == i)
        if mask_i.sum() == 0:
            continue  # skip classes with no samples in this batch

        # average distance from each sample to centroid
        dist_intra = ((embed_code_flat[mask_i] - class_centroids[i]) ** 2).mean()

        intra_class_loss += F.relu(dist_intra - margin_intra)

    # inter class distances
    # compare centroids pairwise
    for i in range(num_classes):
        if not valid_classes[i]:
            continue
        for j in range(i + 1, num_classes):
            if not valid_classes[j]:
                continue

            dist_inter = ((class_centroids[i] - class_centroids[j]) ** 2).mean()
            inter_class_loss += F.relu(margin_inter - dist_inter)

    total_loss = intra_class_loss + inter_class_loss
    return total_loss

def get_ramped_gamma(epoch, base_gamma, ramp_start=0, ramp_epochs=10):
    """
    Linearly ramps gamma ratio for training stability.
    """
    if epoch < ramp_start:
        # before ramp start: gamma = 0
        return 0.0
    elif epoch >= ramp_start + ramp_epochs:
        # past the ramp window, full gamma value
        return base_gamma
    else:
        # scales linearly during ramp
        progress = (epoch - ramp_start) / float(ramp_epochs)
        return base_gamma * progress

# training function
def train(
    epoch,
    loader,
    model,
    optimizer,
    scheduler,
    device,
    clip_grad_main=5.0,    
    clip_grad_diverg=1.0, 
    alpha=0.5,
    beta=0.25,
    base_gamma=0.1,
    ramp_start=0,
    ramp_epochs=10
):
    """
    Args:
        epoch (int): current epoch index
        loader (DataLoader): training data loader
        model (nn.Module)
        optimizer (torch.optim.Optimizer)
        scheduler: optional learning rate scheduler
        device (str): 'cpu' or 'cuda'
        clip_grad_main (float): gradient-norm clip for main losses
        clip_grad_diverg (float): gradient-norm clip for divergence loss
        alpha, beta: weighting factors for S-VQVAE losses
        base_gamma (float): maximum gamma for class divergence
        ramp_start, ramp_epochs: when and how fast to ramp gamma
    """

    # get the current ramped-up gamma for this epoch
    gamma_ramped = get_ramped_gamma(epoch, base_gamma, ramp_start, ramp_epochs)

    if dist.is_primary():
        loader = tqdm(loader)

    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    mse_sum, codebook_sum, commit_sum = 0, 0, 0
    class_loss_sum, divergence_sum, total_loss_sum = 0, 0, 0
    mse_n = 0

    model.train()

    for img, label in loader:
        img, label = img.to(device), label.to(device)

        # forward pass
        recon, latent_loss, logits = model(img)
        quant, diff, embed_ind = model.encode(img)
        embed_code = model.quantize.embed_code(embed_ind).permute(0, 3, 1, 2)

        # compute losses
        recon_loss = recon_criterion(recon, img)
        codebook_loss = ((quant.detach() - embed_code) ** 2).mean()
        commitment_loss = beta * ((quant - embed_code.detach()) ** 2).mean()
        class_loss = class_criterion(logits, label)

        # divergence
        raw_diverg_loss = class_divergence_loss(
            quant,
            labels=label,
            num_classes=10,
            device=device,
            margin_intra=0.5,   
            margin_inter=1.5    
        )

        # separate the "main" loss from the divergence part
        main_loss = alpha * (recon_loss + codebook_loss + commitment_loss) + (1 - alpha) * class_loss
        diverg_loss = gamma_ramped * raw_diverg_loss

        # backprop main losses
        optimizer.zero_grad()
        main_loss.backward(retain_graph=True)

        if clip_grad_main is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_main)

        # backprop divergence loss
        diverg_loss.backward()

        if clip_grad_diverg is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_diverg)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # accumulate losses
        total_loss = main_loss.item() + diverg_loss.item()
        batch_size = img.shape[0]

        mse_sum         += recon_loss.item()       * batch_size
        codebook_sum    += codebook_loss.item()    * batch_size
        commit_sum      += commitment_loss.item()  * batch_size
        class_loss_sum  += class_loss.item()       * batch_size
        divergence_sum  += raw_diverg_loss.item()  * batch_size
        total_loss_sum  += total_loss              * batch_size
        mse_n           += batch_size

    return {
        "recon_loss": mse_sum / mse_n,
        "codebook_loss": codebook_sum / mse_n,
        "commitment_loss": commit_sum / mse_n,
        "class_loss": class_loss_sum / mse_n,
        "divergence_loss": divergence_sum / mse_n,
        "total_loss": total_loss_sum / mse_n,
    }

# validation function
def validate(loader, model, device, alpha=0.5, beta=0.25, gamma=0.1):

    model.eval()

    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    mse_sum, codebook_sum, commit_sum = 0, 0, 0
    class_loss_sum, divergence_sum, total_loss_sum = 0, 0, 0
    mse_n = 0

    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)

            recon, latent_loss, logits = model(img)
            quant, diff, embed_ind = model.encode(img)
            embed_code = model.quantize.embed_code(embed_ind).permute(0, 3, 1, 2)

            recon_loss = recon_criterion(recon, img)
            codebook_loss = ((quant.detach() - embed_code) ** 2).mean()
            commitment_loss = beta * ((quant - embed_code.detach()) ** 2).mean()
            class_loss = class_criterion(logits, label)

            raw_diverg_loss = class_divergence_loss(
                quant,
                labels=label,
                num_classes=10,
                device=device,
                margin_intra=0.5,
                margin_inter=1.5
            )

            total_loss = alpha * (recon_loss + codebook_loss + commitment_loss) \
                         + (1 - alpha) * class_loss \
                         + gamma * raw_diverg_loss

            batch_size = img.shape[0]
            mse_sum         += recon_loss.item()       * batch_size
            codebook_sum    += codebook_loss.item()    * batch_size
            commit_sum      += commitment_loss.item()  * batch_size
            class_loss_sum  += class_loss.item()       * batch_size
            divergence_sum  += raw_diverg_loss.item()  * batch_size
            total_loss_sum  += total_loss.item()       * batch_size
            mse_n           += batch_size

    return {
        "recon_loss": mse_sum / mse_n,
        "codebook_loss": codebook_sum / mse_n,
        "commitment_loss": commit_sum / mse_n,
        "class_loss": class_loss_sum / mse_n,
        "divergence_loss": divergence_sum / mse_n,
        "total_loss": total_loss_sum / mse_n,
    }

# main process
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_model", exist_ok=True)

    # log training process
    if dist.is_primary():
        log_file = "logs/training_loss_cd_emb256.csv"
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Train Total Loss", "Train Recon Loss", "Train Codebook Loss",
                "Train Commitment Loss", "Train Class Loss", "Train Divergence Loss",
                "Val Total Loss", "Val Recon Loss", "Val Codebook Loss", 
                "Val Commitment Loss", "Val Class Loss", "Val Divergence Loss"
            ])

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # normalization
    ])

    dataset = datasets.CIFAR10(root=args.path, train=True, download=True, transform=transform)
    targets = dataset.targets  

    # stratified 80/20 split
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=targets, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # instantiate model
    model = S_VQVAE().to(device)

    # separate weight-decay and no-decay for unsuitable and unnecessary weights for decay
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if "weight" in name and "bn" not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = optim.Adam(
        [{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    # optional scheduler
    scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(train_loader) * args.epoch) \
                if args.sched == "cycle" else None

    best_val_loss = float('inf')
    
    # patience counter for early stopping
    patience = 100  
    patience_counter = 0

    losses = {
        "recon_loss": [],
        "codebook_loss": [],
        "commitment_loss": [],
        "class_loss": [],
        "divergence_loss": [],
        "total_loss": [],
    }
    

    for epoch in range(args.epoch):
        # train
        train_loss = train(
            epoch=epoch,
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            clip_grad_main=args.clip_grad_main,
            clip_grad_diverg=args.clip_grad_diverg,
            alpha=args.alpha,
            beta=args.beta,
            base_gamma=args.gamma,
            ramp_start=args.ramp_start,
            ramp_epochs=args.ramp_epochs
        )

        # validate
        val_loss = validate(
            loader=val_loader,
            model=model,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )

        if dist.is_primary():
            for key in losses.keys():
                losses[key].append(train_loss[key])
            
            # log to CSV
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss["total_loss"], train_loss["recon_loss"], train_loss["codebook_loss"],
                    train_loss["commitment_loss"], train_loss["class_loss"], train_loss["divergence_loss"],
                    val_loss["total_loss"], val_loss["recon_loss"], val_loss["codebook_loss"],
                    val_loss["commitment_loss"], val_loss["class_loss"], val_loss["divergence_loss"]
                ])

            print(
                f"Epoch {epoch}: "
                f"Recon Loss: {train_loss['recon_loss']:.4f}, "
                f"Codebook Loss: {train_loss['codebook_loss']:.4f}, "
                f"Commitment Loss: {train_loss['commitment_loss']:.4f}, "
                f"Class Loss: {train_loss['class_loss']:.4f}, "
                f"Divergence Loss: {train_loss['divergence_loss']:.4f}, "
                f"Total Loss: {train_loss['total_loss']:.4f}"
            )

            # check patience for early stopping
            if val_loss["total_loss"] < best_val_loss:
                best_val_loss = val_loss["total_loss"]
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"best_model/svqvae_cd_emb256.pt"
                )
                print(f"New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print("Early stopping..")
                break

    # plot training loss
    if dist.is_primary():
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(losses["recon_loss"]) + 1),      losses["recon_loss"],      label="Recon Loss",       linestyle="dotted")
        plt.plot(range(1, len(losses["codebook_loss"]) + 1),   losses["codebook_loss"],   label="Codebook Loss",    linestyle="dotted")
        plt.plot(range(1, len(losses["commitment_loss"]) + 1), losses["commitment_loss"], label="Commitment Loss",  linestyle="dotted")
        plt.plot(range(1, len(losses["class_loss"]) + 1),      losses["class_loss"],      label="Class Loss",       linestyle="dotted")
        plt.plot(range(1, len(losses["divergence_loss"]) + 1), losses["divergence_loss"], label="Divergence Loss",  linestyle="dashed")
        plt.plot(range(1, len(losses["total_loss"]) + 1),      losses["total_loss"],      label="Total Loss",       linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Margin-based Class Divergence Model: (Alpha={args.alpha}, Beta={args.beta}, Gamma={args.gamma})")
        plt.legend()
        plt.grid(True)
        plt.savefig("logs/training_plot_cd_emb256.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:12345")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for S-VQVAE losses vs classification")
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight factor")
    parser.add_argument("--gamma", type=float, default=0.1, help="Max gamma for margin-based divergence penalty")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str, help="Set to 'cycle' if using CycleScheduler")
    parser.add_argument("--path", type=str, help="Path to dataset")
    parser.add_argument("--clip_grad_main", type=float, default=5.0, help="Gradient clip norm for main S-VQVAE + classification losses")
    parser.add_argument("--clip_grad_diverg", type=float, default=1.0, help="Gradient clip norm for the divergence term")
    parser.add_argument("--ramp_start", type=int, default=0, help="Epoch at which to begin ramping gamma from 0")
    parser.add_argument("--ramp_epochs", type=int, default=10, help="How many epochs to go from gamma=0 to gamma=(--gamma)")

    args = parser.parse_args()
    main(args)
