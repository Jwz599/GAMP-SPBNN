from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import numpy as np
from networks import *
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--div', default=1.0, type=float, help='diversity_rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--dirsave_out', default='Pruned_GAMP', type=str,
                        help='where the checkpoint are save. ./checkpoint/dataset/dirsave_out')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--use_gamp', action='store_true', help='Use GAMP for latent parameterization')
    parser.add_argument('--use_pruning', action='store_true', help='Use pruning to improve performance')
    parser.add_argument('--pruning_rate', default=0.3, type=float, help='Initial pruning rate')
    parser.add_argument('--final_pruning_rate', default=0.5, type=float, help='Final pruning rate')
    parser.add_argument('--prune_epoch_start', default=50, type=int, help='Epoch to start pruning')
    parser.add_argument('--pruning_warmup', default=10, type=int, help='Warmup epochs before pruning')

    # Add new GPU device parameter
    parser.add_argument('--gpu_id', default='1', type=str, help='GPU device ID to use (default: cuda:1)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    mu_div = args.div
    ensemble_size = 4

    # Pruning configuration
    pruning_config = {
        'pruning_rate': args.pruning_rate,
        'prune_epoch_start': args.prune_epoch_start,
        'final_pruning_rate': args.final_pruning_rate,
        'pruning_warmup': args.pruning_warmup
    }

    # Device setup - modified to use specified GPU device
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set default device to cuda:1
    if use_cuda:
        # Check if the requested GPU is available
        requested_gpu_id = int(args.gpu_id)
        if requested_gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {requested_gpu_id} not available. Using GPU 0 instead.")
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{requested_gpu_id}')

        print(f"Using device: {device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            status = " [SELECTED]" if i == requested_gpu_id else ""
            print(f"GPU {i}: {gpu_name}{status}")
    else:
        device = torch.device('cpu')
        print("Using CPU for training")

    best_acc = 0
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

    # Data Uplaod
    cutout = 16

    class CutoutDefault(object):
        """
        Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
        """

        def __init__(self, length):
            self.length = length

        def __call__(self, img):
            if self.length <= 0:
                return img
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
            return img

    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        CutoutDefault(cutout)
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 100

    print('batch_size=', batch_size)
    # Fix: set num_workers=0 to avoid multiprocessing issues
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)

    # Return network & file name
    def getNetwork(args):
        if args.use_pruning:
            from networks.Wide_ResNet_GAMP_SPBNN import PrunedWideResNetGAMP
            net = PrunedWideResNetGAMP(args.depth, args.widen_factor, args.dropout,
                                       num_classes, num_models=ensemble_size,
                                       use_gamp=args.use_gamp, pruning_config=pruning_config)
            file_name = 'pruned-wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
            if args.use_gamp:
                file_name += '-gamp'
            file_name += f'-prune{args.pruning_rate}'
        else:
            from networks.Wide_ResNet_GAMP_SPBNN import Wide_ResNet_GAMP_SPBNN
            net = Wide_ResNet_GAMP_SPBNN(args.depth, args.widen_factor, args.dropout,
                                         num_classes, num_models=ensemble_size, use_gamp=args.use_gamp)
            file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
            if args.use_gamp:
                file_name += '-gamp'
        return net, file_name

    # Test only option
    if (args.testOnly):
        print('\n[Test Phase] : Model setup')
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        _, file_name = getNetwork(args)
        checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
        net = checkpoint['net']

        # Use specified device during testing as well
        if use_cuda:
            net = net.to(device)
            cudnn.benchmark = True

        net.eval()
        net.training = False
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            acc = 100. * correct / total
            print("| Test Result\tAcc@1: %.2f%%" % (acc))

        sys.exit(0)

    # Model
    print('\n[Phase 2] : Model setup')
    if args.resume:
        # Load checkpoint
        print('| Resuming from checkpoint...')
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        _, file_name = getNetwork(args)
        checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        # Restore pruning state
        if args.use_pruning and 'pruning_stats' in checkpoint:
            net.pruning_stats = checkpoint['pruning_stats']
            net.current_epoch = checkpoint['epoch']
            print(f"Resumed pruning stats: {net.pruning_stats}")
    else:
        print('| Building net type [wide-resnet LB-BNN]...')
        if args.use_pruning:
            print(f"| Using pruning with rate {args.pruning_rate} -> {args.final_pruning_rate}")
            print(f"| Pruning starts at epoch {args.prune_epoch_start}")
        net, file_name = getNetwork(args)

    # Move model to specified device
    if use_cuda:
        net = net.to(device)
        cudnn.benchmark = True
        print(f"Model moved to {device}")
    else:
        print("Model running on CPU")

    criterion = nn.CrossEntropyLoss()

    # Training
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    my_list = ['alpha', 'gamma']

    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    def create_pruning_optimizer(model, lr, epoch):
        """Create optimizer for pruned model"""
        if not args.use_pruning:
            # Original optimizer setup
            params_multi_tmp = list(
                filter(lambda kv: ('alpha' in kv[0]) or ('gamma' in kv[0]), model.named_parameters()))
            param_core_tmp = list(
                filter(lambda kv: ('alpha' not in kv[0]) and ('gamma' not in kv[0]), model.named_parameters()))
            params_multi = [param for name, param in params_multi_tmp]
            param_core = [param for name, param in param_core_tmp]

            optimizer = optim.SGD([
                {'params': param_core, 'weight_decay': 5e-4},
                {'params': params_multi, 'weight_decay': 0.0}
            ], lr=cf.learning_rate(lr, epoch), momentum=0.9)
        else:
            # Optimizer setup for pruned model
            pruned_params = []
            normal_params = []
            importance_params = []
            mask_params = []

            for name, param in model.named_parameters():
                if 'importance' in name or 'mask' in name:
                    mask_params.append(param)
                elif 'alpha' in name or 'gamma' in name:
                    pruned_params.append(param)
                else:
                    normal_params.append(param)

            optimizer = optim.SGD([
                {'params': normal_params, 'weight_decay': 5e-4, 'lr': cf.learning_rate(lr, epoch)},
                {'params': pruned_params, 'weight_decay': 0.0, 'lr': cf.learning_rate(lr, epoch)},
                {'params': mask_params, 'weight_decay': 0.0, 'lr': cf.learning_rate(lr, epoch) * 0.1}
            ], momentum=0.9)

        return optimizer

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        # Create optimizer
        optimizer = create_pruning_optimizer(net, args.lr, epoch)

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, cf.learning_rate(args.lr, epoch)))

        # Update importance scores (if in pruning mode)
        if args.use_pruning and epoch >= args.prune_epoch_start:
            net.update_importance_scores()

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs = torch.cat([inputs for i in range(ensemble_size)], dim=0)
            targets = torch.cat([targets for i in range(ensemble_size)], dim=0)

            # Use specified device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_cuda):
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)

                # Calculate base loss
                if args.use_pruning:
                    base_loss = criterion(outputs, targets)

                    # Add pruning regularization loss
                    pruning_reg = net.get_pruning_regularization()
                    pruning_weight = 0.01 * min(1.0, (
                                epoch - args.prune_epoch_start) / 50) if epoch >= args.prune_epoch_start else 0.0

                    if args.use_gamp:
                        gamp_losses = []
                        for module in net.modules():
                            if hasattr(module, 'loss_latent'):
                                gamp_losses.append(module.loss_latent)

                        if gamp_losses:
                            avg_gamp_loss = sum(gamp_losses) / len(gamp_losses)
                            # Reduce GAMP loss weight to avoid overconfidence
                            gamp_penalty = mu_div * 0.3 * avg_gamp_loss
                            loss = base_loss + gamp_penalty + pruning_weight * pruning_reg
                        else:
                            loss = base_loss + pruning_weight * pruning_reg
                    else:
                        loss = base_loss + mu_div * loss_latent_from_nn(net) + pruning_weight * pruning_reg

                    # Record loss components for debugging
                    if batch_idx % 100 == 0 and epoch >= args.prune_epoch_start:
                        print(f'[Pruning] Base: {base_loss.item():.4f}, Pruning Reg: {pruning_reg.item():.4f}')
                else:
                    # Original loss calculation
                    if args.use_gamp:
                        gamp_losses = []
                        for module in net.modules():
                            if hasattr(module, 'loss_latent'):
                                gamp_losses.append(module.loss_latent)

                        if gamp_losses:
                            avg_gamp_loss = sum(gamp_losses) / len(gamp_losses)
                            loss = criterion(outputs, targets) + mu_div * avg_gamp_loss

                            if batch_idx % 100 == 0:
                                print(f'GAMP loss: {avg_gamp_loss.item():.4f}')
                        else:
                            loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, targets) + mu_div * loss_latent_from_nn(net)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                             % (epoch, num_epochs, batch_idx + 1,
                                (len(trainset) // batch_size) + 1, loss.item(), 100. * correct / total))
            sys.stdout.flush()

        # Apply pruning masks (if in pruning mode)
        if args.use_pruning:
            pruning_applied = net.apply_pruning_masks(epoch)
            if pruning_applied:
                print(f"\n| Pruning applied at epoch {epoch}, ratio: {net.pruning_stats['pruning_ratio'] * 100:.2f}%")

    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # Use specified device
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            # Save checkpoint when best model
            acc = 100. * correct / total
            print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))

            # Show pruning statistics if in pruning mode
            if args.use_pruning:
                print(
                    f"| Pruning Stats: {net.pruning_stats['remaining_parameters']}/{net.pruning_stats['total_parameters']} "
                    f"({net.pruning_stats['pruning_ratio'] * 100:.2f}% pruned)")

            if acc > best_acc:
                print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
                state = {
                    'net': net,
                    'acc': acc,
                    'epoch': epoch,
                }
                # Save pruning statistics
                if args.use_pruning:
                    state['pruning_stats'] = net.pruning_stats

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                save_point = './checkpoint/' + args.dataset + os.sep
                if not os.path.isdir(save_point): os.mkdir(save_point)
                save_point = save_point + args.dirsave_out + os.sep
                if not os.path.isdir(save_point): os.mkdir(save_point)
                torch.save(state, save_point + file_name + '.t7')
                best_acc = acc

    # Print training configuration
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))
    print('| Device = ' + str(device))
    if args.use_pruning:
        print('| Pruning Configuration:')
        print(f'|   - Pruning Rate: {args.pruning_rate} -> {args.final_pruning_rate}')
        print(f'|   - Start Epoch: {args.prune_epoch_start}')
        print(f'|   - Warmup Epochs: {args.pruning_warmup}')

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        train(epoch)
        test(epoch)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

    # Final pruning statistics
    if args.use_pruning:
        print('\n[Final Pruning Statistics]')
        print(f"Total Parameters: {net.pruning_stats['total_parameters']}")
        print(f"Remaining Parameters: {net.pruning_stats['remaining_parameters']}")
        print(f"Pruning Ratio: {net.pruning_stats['pruning_ratio'] * 100:.2f}%")
        print(f"Compression Ratio: {1 / (1 - net.pruning_stats['pruning_ratio']):.2f}x")


if __name__ == '__main__':
    # Fix multiprocessing issues
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    main()