import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scheduler import CycleScheduler
from models.cnn_classifier import CNNClassifier 

# adversarial perturbation PGD-k
def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, n_iter=10):
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad_(True)  
    
    for _ in range(n_iter):
        output = model(perturbed_images)
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
def train(epoch, loader, model, optimizer, scheduler, device, use_adversarial=False):
    loader = tqdm(loader)
    criterion = nn.CrossEntropyLoss()
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        if use_adversarial:
            images = pgd_attack(model, images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)

# validation function
def validate(loader, model, device, use_adversarial=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_adversarial:
                images.requires_grad = True
                with torch.enable_grad():
                    images = pgd_attack(model, images, labels)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# main process
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_model", exist_ok=True)
    
    # log training process
    log_file = "logs/training_loss_cnn_classifier_delayedat.csv"

    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Acc", "Adv Val Loss", "Adv Val Acc"])

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.CIFAR10(root=args.path, train=True, download=True, transform=transform)
    targets = dataset.targets  
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, stratify=targets, random_state=42)
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # instantiate model
    model = CNNClassifier().to(device)
    
    # weight-decay only for necessary parameters
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
    scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(train_loader) * args.epoch) if args.sched == "cycle" else None
    
    best_val_loss = float('inf')
    best_combined_val_loss = float('inf')
    
    # patience counter
    patience, patience_counter = 100, 0
    
    moving_avg_loss, switch_point_detected = None, False
    
    for epoch in range(args.epoch):
        train_loss = train(epoch, train_loader, model, optimizer, scheduler, device, use_adversarial=switch_point_detected)
        val_loss, val_acc = validate(val_loader, model, device, use_adversarial=False)
        adv_val_loss, adv_val_acc = validate(val_loader, model, device, use_adversarial=True) if switch_point_detected else (None, None)
        combined_val_loss = val_loss + (adv_val_loss if adv_val_loss else 0)
        
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc, adv_val_loss, adv_val_acc])
        
        if switch_point_detected:
            if combined_val_loss < best_combined_val_loss:
                best_combined_val_loss = combined_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model/cnn_classifier_delayedat.pth")
                print(f"Saving best model at epoch {epoch+1} with val_loss {val_loss:.4f} | adv_val_loss {adv_val_loss:.4f} | combined {combined_val_loss}")
            else:
                patience_counter += 1
        
        if not switch_point_detected:
            if moving_avg_loss is None:
                moving_avg_loss = val_loss
            else:
                moving_avg_loss = 0.9 * moving_avg_loss + 0.1 * val_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model/cnn_classifier_delayedat.pth")
                print(f"Saving new best model at epoch {epoch+1} with val_loss {val_loss:.4f}")
                
            if val_loss > moving_avg_loss:
                switch_point_detected = True
                print(f"Switching to adversarial training at epoch {epoch+1}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}...")
            break

# parser to adjust to desired training process
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--path", type=str, default="./data")
    parser.add_argument("--sched", type=str)
    args = parser.parse_args()
    main(args)
