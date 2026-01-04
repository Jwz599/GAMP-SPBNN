from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import numpy as np
from torch.autograd import Variable
from metric_OOD import eval_ood_measure
from sklearn.metrics import classification_report, confusion_matrix

# 1. Negative Log-Likelihood (NLL) calculation function
def calculate_nll(mean_probs, targets):
    eps = 1e-10  # Prevent log(0)
    log_probs = torch.log(mean_probs + eps)
    nll = -torch.gather(log_probs, dim=1, index=targets.unsqueeze(1)).mean()
    return nll.item()

# 2. Fixed ECE calculation class
class ECELossFixed(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELossFixed, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

# 3. Confusion matrix visualization function
def plot_confusion_matrix(y_true, y_pred, classes, save_path, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10) if len(classes) == 100 else (10, 8))
    sns.heatmap(cm, annot=False if len(classes) == 100 else True,
                fmt="d", cmap="Blues",
                xticklabels=classes[:10] if len(classes) == 100 else classes,
                yticklabels=classes[:10] if len(classes) == 100 else classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # plt.title(f"Confusion Matrix (GAMP-SPBNN, {dataset_name})")
    # plt.title(f"Confusion Matrix (SPBNN, {dataset_name})")
    plt.title(f"Confusion Matrix (GAMP-BNN, {dataset_name})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 4. Classification report saving function
def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")

# Command line argument parsing
parser = argparse.ArgumentParser(description='PyTorch CIFAR GAMP-BNN Evaluation')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate (for compatibility)')
parser.add_argument('--dirsave_out', default='./checkpoint/cifar10/GAMP_SPBNN',
                    type=str, help='checkpoint save directory')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model type (for compatibility)')
parser.add_argument('--depth', default=28, type=int, help='WideResNet depth')
parser.add_argument('--widen_factor', default=10, type=int, help='WideResNet width factor')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate (for compatibility)')
parser.add_argument('--algo', default='GAMP_SPBNN', type=str, help='algorithm name: GAMP_BNN/SPBNN')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'],
                    help='dataset to evaluate')
parser.add_argument('--use_gamp', action='store_true', default=True, help='use GAMP module (for compatibility)')
parser.add_argument('--num_models', default=1, type=int, help='number of models to evaluate')
parser.add_argument('--ensemble_size', default=4, type=int, help='ensemble model count (must match training)')
args = parser.parse_args()

# ===================== Core Modification 1: Specify cuda:1 as default device =====================
use_cuda = torch.cuda.is_available()
# Prefer to use cuda:1, fallback to other available GPUs if cuda:1 is unavailable
if use_cuda:
    # Check if cuda:1 is available
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0')  # Use cuda:0 for single GPU setup
    # Clear cache of cuda:1 to release redundant video memory
    torch.cuda.empty_cache()
    # Specify cuda:1 for subsequent calculations
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')
ensemble_size = args.ensemble_size
print(f"Using device: {device}")  # Print current device to verify if it's cuda:1

# Dataset adaptation: Dynamically set number of classes and class names
if args.dataset == 'cifar10':
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ood_label = 10  # Set OOD sample label to ID class count + 1
elif args.dataset == 'cifar100':
    num_classes = 100
    # CIFAR-100 official class names (can be replaced as needed)
    classes = [f'class_{i}' for i in range(num_classes)]
    ood_label = 100  # Adapt OOD label for 100 classes
else:
    raise ValueError("Dataset must be 'cifar10' or 'cifar100'")

# Data preprocessing: Dynamically load mean/standard deviation based on dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

# Load ID dataset (CIFAR-10/CIFAR-100)
print(f"| Preparing {args.dataset.upper()} dataset...")
if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
else:
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
# ===================== Optimization 1: Reduce batch size to further lower memory usage =====================
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# Load OOD dataset (SVHN)
print("| Preparing SVHN OOD dataset...")
testset_OOD = torchvision.datasets.SVHN(root='./data', split='test',
                                        transform=transform_test, download=True)
# ===================== Optimization 1: Reduce batch size for OOD dataset =====================
testloader_OOD = torch.utils.data.DataLoader(testset_OOD, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# Model loading function
def getNetwork(args):
    if args.algo not in ['GAMP_SPBNN', 'SPBNN']:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    from networks.Wide_ResNet_GAMP_SPBNN import Wide_ResNet_GAMP_SPBNN
    net = Wide_ResNet_GAMP_SPBNN(
        depth=args.depth,
        widen_factor=args.widen_factor,
        dropout_rate=args.dropout,
        num_classes=num_classes,
        num_models=ensemble_size,
        use_gamp=args.use_gamp
    )
    file_name = f'wide-resnet-{args.depth}x{args.widen_factor}'
    name_algo = 'GAMP-BNN'
    return net, file_name, name_algo

# Load model
net, file_name, name_algo = getNetwork(args)
print(f"| Loaded model: {file_name}")

# Ensemble inference function (adapted to GAMP-BNN forward logic)
def test_ensemble(model, testloader, return_uncertainty=False):
    model.eval()
    all_mean_probs = []
    all_targets = []
    all_predictions = []
    all_uncertainty = []

    with torch.no_grad():  # Disable gradient calculation to significantly reduce memory usage
        for inputs, targets in testloader:
            inputs = inputs.to(device, non_blocking=True)  # non_blocking=True improves efficiency without increasing memory
            targets = targets.to(device, non_blocking=True)

            # Call GAMP-BNN forward (no return_uncertainty parameter)
            output = model(inputs)
            mean_probs = output  # GAMP-BNN directly returns weighted average probability in test phase
            predictions = torch.argmax(mean_probs, dim=1)

            if return_uncertainty:
                # Manually calculate uncertainty metrics (adapted to original logic)
                batch_size = inputs.shape[0]
                # Reshape to ensemble model format [num_models, batch_size, num_classes]
                out_reshaped = mean_probs.repeat(ensemble_size, 1, 1).view(ensemble_size, batch_size, num_classes)
                individual_probs = out_reshaped
                max_probs = torch.max(mean_probs, dim=1)[0]
                individual_preds = torch.argmax(individual_probs, dim=-1)
                disagreement = torch.std(individual_preds.float(), dim=0)
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)

                uncertainty_info = {
                    'mean_probs': mean_probs,
                    'max_prob': max_probs,
                    'predictions': predictions,
                    'disagreement': disagreement,
                    'entropy': entropy,
                    'ensemble_variance': torch.var(individual_probs, dim=0).mean(dim=-1)
                }
                all_uncertainty.append(uncertainty_info)

            # Move tensors to CPU first before storage to avoid GPU memory accumulation
            all_mean_probs.append(mean_probs.cpu())
            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())
            # ===================== Optimization 2: Clear useless tensors after each batch to release memory =====================
            del inputs, targets, output, mean_probs, predictions
            if return_uncertainty:
                del out_reshaped, individual_probs, max_probs, individual_preds, disagreement, entropy
            torch.cuda.empty_cache()

    mean_probs_cat = torch.cat(all_mean_probs, dim=0)
    predictions_cat = torch.cat(all_predictions, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    if return_uncertainty:
        return mean_probs_cat, predictions_cat, targets_cat, all_uncertainty[0]  # Return uncertainty of the first batch
    return mean_probs_cat, predictions_cat, targets_cat

# Metric storage container
nll_list = []
auroc_list = []
aupr_list = []
fpr_list = []
ece_list = []
acc_list = []

# Multi-model evaluation loop
for step in range(args.num_models):
    print(f"\n{'=' * 60}")
    print(f"Evaluating model {step + 1}/{args.num_models}")
    print(f"{'=' * 60}")

    # Load model weights
    checkpoint_path = os.path.join(args.dirsave_out + str(step), file_name + '.t7')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ===================== Core Modification 2: Specify cuda:1 when loading model =====================
    checkpoint = torch.load(checkpoint_path, map_location=device)  # map_location=device loads directly to specified device (cuda:1)
    net = checkpoint['net'].to(device)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # 1. Evaluate ID dataset (CIFAR-10/CIFAR-100)
    mean_probs, predictions, targets, _ = test_ensemble(net, testloader, return_uncertainty=True)

    # Calculate classification accuracy
    acc = (predictions == targets).float().mean().item()
    acc_list.append(acc)
    print(f"\nID Dataset Metrics:")
    print(f"Accuracy: {acc:.4f}")

    # Calculate NLL
    nll = calculate_nll(mean_probs, targets)
    nll_list.append(nll)
    print(f"NLL: {nll:.4f}")

    # Calculate ECE
    confidences = torch.max(mean_probs, dim=1)[0]
    ece_calculator = ECELossFixed(n_bins=20 if num_classes == 100 else 15)
    # Move to specified device for ECE calculation
    ece = ece_calculator(confidences.to(device), predictions.to(device), targets.to(device)).item()
    ece_list.append(ece)
    print(f"ECE: {ece:.6f}")

    # Generate confusion matrix and classification report
    plot_confusion_matrix(
        targets.numpy(), predictions.numpy(),
        classes,
        save_path=f'confusion_matrix_{name_algo}_{args.dataset}_model{step + 1}.png',
        dataset_name=args.dataset.upper()
    )
    save_classification_report(
        targets.numpy(), predictions.numpy(),
        classes,
        save_path=f'classification_report_{name_algo}_{args.dataset}_model{step + 1}.txt'
    )

    # 2. Evaluate OOD dataset (SVHN)
    ood_mean_probs, ood_predictions, ood_targets = test_ensemble(net, testloader_OOD)
    ood_confidences = torch.max(ood_mean_probs, dim=1)[0]

    # 3. Calculate OOD detection metrics (adapted to dual dataset labels)
    conf = torch.cat([ood_confidences.to(device), confidences.to(device)], dim=0)
    label = torch.cat([
        ood_label * torch.ones_like(ood_targets).long().to(device),
        targets.long().to(device)
    ], dim=0)
    pred = torch.cat([
        torch.zeros_like(ood_predictions).long().to(device),
        predictions.long().to(device)
    ], dim=0)

    auroc, aupr, fpr, _ = eval_ood_measure(conf, label, pred)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print(f"\nOOD Detection Metrics:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR@95TPR: {fpr:.4f}")

    # ===================== Optimization 3: Clear memory after each model evaluation =====================
    del net, mean_probs, predictions, targets, ood_mean_probs, ood_predictions, ood_targets
    del conf, label, pred
    torch.cuda.empty_cache()

# Final metric summary
print(f"\n{'=' * 60}")
print(f"Final Evaluation Metrics (Mean over {args.num_models} models)")
print(f"{'=' * 60}")
print(f"Dataset: {args.dataset.upper()}")
print(f"Model: {name_algo} (WideResNet-{args.depth}x{args.widen_factor})")
print(f"Ensemble Size: {ensemble_size}")
print(f"{'=' * 60}")
print(f"Classification Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"Mean NLL: {np.mean(nll_list):.4f} ± {np.std(nll_list):.4f}")
print(f"Mean ECE: {np.mean(ece_list):.6f} ± {np.std(ece_list):.6f}")
print(f"Mean AUROC (OOD): {np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}")
print(f"Mean AUPR (OOD): {np.mean(aupr_list):.4f} ± {np.std(aupr_list):.4f}")
print(f"Mean FPR@95TPR (OOD): {np.mean(fpr_list):.4f} ± {np.std(fpr_list):.4f}")
print(f"{'=' * 60}")