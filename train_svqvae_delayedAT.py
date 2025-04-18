"""
DelayedAT adversarial training process for S-VQVAE
"""

import argparse
import os
import csv
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import distributed as dist
from models.s_vqvae_emb256 import S_VQVAE   ### change model accordingly
from scheduler import CycleScheduler
from sklearn.model_selection import train_test_split

# adversarial perturbation PGD-k
def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, n_iter=10):
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad_(True)  
    
    for _ in range(n_iter):
        output = model(perturbed_images)[2]  # Logits
        loss = F.cross_entropy(output, labels)
        
        model.zero_grad()  
        loss.backward()

        with torch.no_grad():
            perturbed_images += alpha * perturbed_images.grad.sign()
            perturbation = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(images + perturbation, min=-1, max=1).detach()
            perturbed_images.requires_grad_(True)  

    return perturbed_images

# training function
def train(epoch, loader, model, optimizer, scheduler, device, clip_grad=5.0, alpha=0.5, beta=0.25, use_adversarial=False):
    if dist.is_primary():
        loader = tqdm(loader)

    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    mse_sum, codebook_sum, commit_sum, class_loss_sum, total_loss_sum = 0, 0, 0, 0, 0
    mse_n = 0

    model.train()

    for img, label in loader:
        img, label = img.to(device), label.to(device)

        if use_adversarial:
            img = pgd_attack(model, img, label)
        
        # forward pass
        recon, latent_loss, logits = model(img)
        quant, diff, embed_ind = model.encode(img)
        embed_code = model.quantize.embed_code(embed_ind).permute(0, 3, 1, 2)

        # compute losses
        recon_loss = recon_criterion(recon, img)
        codebook_loss = ((quant.detach() - embed_code) ** 2).mean()
        commitment_loss = beta * ((quant - embed_code.detach()) ** 2).mean()
        class_loss = class_criterion(logits, label)

        total_loss = alpha * (recon_loss + codebook_loss + commitment_loss) + (1 - alpha) * class_loss

        optimizer.zero_grad()
        total_loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # accumulate losses
        mse_sum += recon_loss.item() * img.shape[0]
        codebook_sum += codebook_loss.item() * img.shape[0]
        commit_sum += commitment_loss.item() * img.shape[0]
        class_loss_sum += class_loss.item() * img.shape[0]
        total_loss_sum += total_loss.item() * img.shape[0]
        mse_n += img.shape[0]

    return {
        "recon_loss": mse_sum / mse_n,
        "codebook_loss": codebook_sum / mse_n,
        "commitment_loss": commit_sum / mse_n,
        "class_loss": class_loss_sum / mse_n,
        "total_loss": total_loss_sum / mse_n,
    }

# validation function
def validate(loader, model, device, use_adversarial=False):
    model.eval()
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    mse_sum, codebook_sum, commit_sum, class_loss_sum, total_loss_sum = 0, 0, 0, 0, 0
    mse_n = 0
    
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            
            if use_adversarial:
                img.requires_grad = True
                with torch.enable_grad():
                    img = pgd_attack(model, img, label)
            
            recon, latent_loss, logits = model(img)
            quant, diff, embed_ind = model.encode(img)
            embed_code = model.quantize.embed_code(embed_ind).permute(0, 3, 1, 2)

            recon_loss = recon_criterion(recon, img)
            codebook_loss = ((quant.detach() - embed_code) ** 2).mean()
            commitment_loss = ((quant - embed_code.detach()) ** 2).mean()
            class_loss = class_criterion(logits, label)
            
            total_loss = recon_loss + codebook_loss + commitment_loss + class_loss
            
            mse_sum += recon_loss.item() * img.shape[0]
            codebook_sum += codebook_loss.item() * img.shape[0]
            commit_sum += commitment_loss.item() * img.shape[0]
            class_loss_sum += class_loss.item() * img.shape[0]
            total_loss_sum += total_loss.item() * img.shape[0]
            mse_n += img.shape[0]
    
    return {
        "recon_loss": mse_sum / mse_n,
        "codebook_loss": codebook_sum / mse_n,
        "commitment_loss": commit_sum / mse_n,
        "class_loss": class_loss_sum / mse_n,
        "total_loss": total_loss_sum / mse_n,
    }

# main process
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_model", exist_ok=True)

    # log training process
    log_file = "logs/training_loss_delayedat_emb256.csv"

    if dist.is_primary():
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Total Loss", "Train Recon Loss", "Train Codebook Loss", "Train Commitment Loss", "Train Class Loss",
                             "Val Total Loss", "Val Recon Loss", "Val Codebook Loss", "Val Commitment Loss", "Val Class Loss",
                             "Adv Val Total Loss", "Adv Val Recon Loss", "Adv Val Codebook Loss", "Adv Val Commitment Loss", "Adv Val Class Loss"])


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

    # weight-decay only for necessary params
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
    scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(train_loader) * args.epoch) if args.sched == "cycle" else None

    best_val_loss = float('inf')
    best_combined_val_loss = float('inf')
    
    # patience counter
    patience = 100
    patience_counter = 0
    
    moving_avg_loss, switch_point_detected = None, False
    
    losses = {"recon_loss": [], "codebook_loss": [], "commitment_loss": [], "class_loss": [], "total_loss": []}

    for epoch in range(args.epoch):
        train_loss = train(epoch, train_loader, model, optimizer, scheduler, device, alpha=args.alpha, beta=args.beta, use_adversarial=switch_point_detected)
        val_loss = validate(val_loader, model, device, use_adversarial=False)
        adv_val_loss = validate(val_loader, model, device, use_adversarial=True) if switch_point_detected else None
        combined_val_loss = val_loss["total_loss"] + (adv_val_loss["total_loss"] if adv_val_loss else 0)
        
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1] + list(train_loss.values()) + list(val_loss.values()) + (list(adv_val_loss.values()) if adv_val_loss else [None] * 5))
        
        if switch_point_detected:
            if combined_val_loss < best_combined_val_loss:
                best_combined_val_loss = combined_val_loss
                patience_counter = 0
                print(f"Saving new best model: val_loss: {val_loss["total_loss"]:.4f} | adv_val_loss: {adv_val_loss["total_loss"]:.4f} | combined: {combined_val_loss:.4f}")
                torch.save(model.state_dict(), f"best_model/svqvae_delayedat_emb256.pt")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
                
        
        if not switch_point_detected:
            if moving_avg_loss is None:
                moving_avg_loss = val_loss["total_loss"]
            else:
                moving_avg_loss = 0.9 * moving_avg_loss + 0.1 * val_loss["total_loss"]

            if val_loss["total_loss"] < best_val_loss:
                best_val_loss = val_loss["total_loss"]
                print(f"Saving new best model: val_loss {val_loss["total_loss"]:.4f}")
                torch.save(model.state_dict(), f"best_model/svqvae_delayedat_emb256.pt")
                
            if val_loss["total_loss"] > moving_avg_loss:   # switch to adversarial training at the start of convergence
                switch_point_detected = True
                print(f"Switching to adversarial training at epoch {epoch+1}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}...")
            break
    

    # plot training loss
    if dist.is_primary():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses["recon_loss"]) + 1), losses["recon_loss"], label="Recon Loss", linestyle="dotted", color="blue")
        plt.plot(range(1, len(losses["codebook_loss"]) + 1), losses["codebook_loss"], label="Codebook Loss", linestyle="dotted", color="green")
        plt.plot(range(1, len(losses["commitment_loss"]) + 1), losses["commitment_loss"], label="Commitment Loss", linestyle="dotted", color="purple")
        plt.plot(range(1, len(losses["class_loss"]) + 1), losses["class_loss"], label="Class Loss", linestyle="dotted", color="orange")

        plt.plot(range(1, len(losses["total_loss"]) + 1), losses["total_loss"], label="Total Loss", linewidth=2, color="red")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Components DelayedAT Model: (Alpha={args.alpha}, Beta={args.beta})")
        plt.legend()
        plt.grid(True)
        plt.savefig("logs/training_plot_delayedat_emb256.png")
        plt.show()

# parser to adjust to desired training process
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:12345")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    main(args)
