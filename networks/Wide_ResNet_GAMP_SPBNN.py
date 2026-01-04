import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from networks.GAMP_SPBNN_layers import *  # Modify the import path

import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, first_layer=False, num_models=4, use_gamp=True, pruning_rate=0.3):
    return PrunedEnsemble_Conv2d_GAMP(in_planes, out_planes, 3, stride=stride, padding=1,
                                      first_layer=first_layer, num_models=num_models,
                                      use_gamp=use_gamp, pruning_rate=pruning_rate)


def loss_latent_from_nn(model):
    """Gathers the KL Divergence from a nn.Module object"""
    loss_latent = 0
    loss_latent_inc = 1
    for module in model.modules():
        if isinstance(module, (EnsembleModule)):
            loss_latent += module.loss_latent
            loss_latent_inc += 1
    return loss_latent / loss_latent_inc


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            init.xavier_uniform_(m.conv.conv.weight, gain=np.sqrt(2))
        except:
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class PrunedWideBasicGAMP(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, num_models=4,
                 use_gamp=True, pruning_rate=0.3):
        super(PrunedWideBasicGAMP, self).__init__()
        self.bn1 = Ensemble_BatchNorm2d(in_planes, num_models=num_models)
        self.conv1 = PrunedEnsemble_Conv2d_GAMP(in_planes, planes, 3, stride=1, padding=1,
                                                first_layer=False, num_models=num_models,
                                                use_gamp=use_gamp, pruning_rate=pruning_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = Ensemble_BatchNorm2d(planes, num_models=num_models)
        self.conv2 = PrunedEnsemble_Conv2d_GAMP(planes, planes, 3, stride=stride, padding=1,
                                                first_layer=False, num_models=num_models,
                                                use_gamp=use_gamp, pruning_rate=pruning_rate)
        self.num_models = num_models
        self.convs = [self.conv1, self.conv2]

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PrunedEnsemble_Conv2d_GAMP(in_planes, planes, 1, stride=stride, padding=0,
                                           first_layer=False, num_models=num_models,
                                           use_gamp=use_gamp, pruning_rate=pruning_rate),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class PrunedWideResNetGAMP(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, num_models,
                 use_gamp=True, pruning_config=None):
        super(PrunedWideResNetGAMP, self).__init__()

        # Pruning configuration
        if pruning_config is None:
            pruning_config = {
                'pruning_rate': 0.3,
                'prune_epoch_start': 50,
                'final_pruning_rate': 0.5
            }
        self.pruning_config = pruning_config
        self.current_epoch = 0

        self.in_planes = 16
        self.num_models = num_models
        self.use_gamp = use_gamp

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Pruned Wide-Resnet %dx%d with GAMP' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0], stride=1, first_layer=True,
                             num_models=num_models, use_gamp=use_gamp,
                             pruning_rate=pruning_config['pruning_rate'])

        self.layer1 = self._wide_layer(PrunedWideBasicGAMP, nStages[1], n,
                                       dropout_rate, stride=1, num_models=num_models,
                                       use_gamp=use_gamp, pruning_rate=pruning_config['pruning_rate'])
        self.layer2 = self._wide_layer(PrunedWideBasicGAMP, nStages[2], n,
                                       dropout_rate, stride=2, num_models=num_models,
                                       use_gamp=use_gamp, pruning_rate=pruning_config['pruning_rate'])
        self.layer3 = self._wide_layer(PrunedWideBasicGAMP, nStages[3], n,
                                       dropout_rate, stride=2, num_models=num_models,
                                       use_gamp=use_gamp, pruning_rate=pruning_config['pruning_rate'])

        self.bn1 = Ensemble_BatchNorm2d(nStages[3], num_models=num_models)
        self.linear = PrunedEnsemble_FC_GAMP(nStages[3], num_classes, False, num_models,
                                             use_gamp=use_gamp, pruning_rate=pruning_config['pruning_rate'])
        self.num_classes = num_classes

        # Pruning statistics
        self.pruning_stats = {
            'total_parameters': 0,
            'remaining_parameters': 0,
            'pruning_ratio': 0.0
        }

        # Initialize pruning state
        self._set_pruning_enabled(False)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, num_models, use_gamp, pruning_rate):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, num_models, use_gamp, pruning_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _set_pruning_enabled(self, enabled):
        """Set whether pruning is enabled for all pruning layers"""
        for module in self.modules():
            if hasattr(module, 'apply_pruning'):
                module.apply_pruning = enabled
            if hasattr(module, 'gamp_module') and hasattr(module.gamp_module, 'apply_pruning'):
                module.gamp_module.apply_pruning = enabled

    def update_importance_scores(self):
        """Update importance scores for all layers"""
        for module in self.modules():
            if hasattr(module, 'update_importance'):
                module.update_importance()

    def apply_pruning_masks(self, epoch):
        """Apply pruning masks according to training progress"""
        self.current_epoch = epoch

        # Check if pruning should start
        if epoch < self.pruning_config['prune_epoch_start']:
            return False

        # Enable pruning
        self._set_pruning_enabled(True)

        # Calculate progressive pruning rate
        progress = min(1.0, (epoch - self.pruning_config['prune_epoch_start']) /
                       (200 - self.pruning_config['prune_epoch_start']))  # Assume total epochs is 200
        current_pruning_rate = self.pruning_config['pruning_rate'] + \
                               (self.pruning_config['final_pruning_rate'] - self.pruning_config[
                                   'pruning_rate']) * progress

        print(f"Epoch {epoch}: Applying pruning with rate {current_pruning_rate:.3f}")

        # Update pruning rate for all layers and apply masks
        total_params = 0
        remaining_params = 0

        for module in self.modules():
            if hasattr(module, 'pruning_rate'):
                module.pruning_rate = current_pruning_rate
            if hasattr(module, 'gamp_module') and hasattr(module.gamp_module, 'pruning_rate'):
                module.gamp_module.pruning_rate = current_pruning_rate

            if hasattr(module, 'apply_pruning_mask'):
                module.apply_pruning_mask(epoch)

            # Statistics collection
            if hasattr(module, 'weight_mask'):
                total_params += module.weight_mask.numel()
                remaining_params += module.weight_mask.sum().item()
            if hasattr(module, 'gamp_module'):
                if hasattr(module.gamp_module, 'theta_mask'):
                    total_params += module.gamp_module.theta_mask.numel()
                    remaining_params += module.gamp_module.theta_mask.sum().item()
                if hasattr(module.gamp_module, 'phi_mask'):
                    total_params += module.gamp_module.phi_mask.numel()
                    remaining_params += module.gamp_module.phi_mask.sum().item()

        # Update statistics
        if total_params > 0:
            self.pruning_stats['total_parameters'] = total_params
            self.pruning_stats['remaining_parameters'] = remaining_params
            self.pruning_stats['pruning_ratio'] = 1 - (remaining_params / total_params)

            print(f"Pruning statistics: {remaining_params}/{total_params} "
                  f"({self.pruning_stats['pruning_ratio'] * 100:.2f}% pruned)")

        return True

    def get_pruning_regularization(self):
        """Calculate pruning regularization loss to encourage sparsity"""
        regularization_loss = 0
        count = 0

        for module in self.modules():
            if hasattr(module, 'weight_importance') and hasattr(module, 'fc'):
                # L1 regularization to encourage sparsity
                regularization_loss += torch.mean(torch.abs(module.fc.weight))
                count += 1
            elif hasattr(module, 'weight_importance') and hasattr(module, 'conv'):
                regularization_loss += torch.mean(torch.abs(module.conv.weight))
                count += 1

        return regularization_loss / count if count > 0 else 0

    def forward(self, x, return_uncertainty=False, return_pruning_stats=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if not self.training:
            # Calculate uncertainty information (if enabled)
            if return_uncertainty:
                ensemble_outputs = out.view([self.num_models, -1, self.num_classes])
                individual_probs = F.softmax(ensemble_outputs, dim=-1)
                mean_probs = torch.mean(individual_probs, dim=0)

                # Calculate uncertainty metrics
                max_probs, predictions = torch.max(mean_probs, dim=-1)
                individual_predictions = torch.argmax(ensemble_outputs, dim=-1)
                disagreement = torch.std(individual_predictions.float(), dim=0)
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)

                uncertainty_info = {
                    'mean_probs': mean_probs,
                    'max_prob': max_probs,
                    'predictions': predictions,
                    'disagreement': disagreement,
                    'entropy': entropy,
                    'ensemble_variance': torch.var(individual_probs, dim=0).mean(dim=-1)
                }

                if return_pruning_stats:
                    return mean_probs, uncertainty_info, self.pruning_stats
                else:
                    return mean_probs, uncertainty_info
            else:
                out = F.softmax(out, dim=1)
                result = out.view([self.num_models, -1, self.num_classes]).mean(dim=0)
                if return_pruning_stats:
                    return result, self.pruning_stats
                else:
                    return result
        else:
            if return_pruning_stats:
                return out, self.pruning_stats
            else:
                return out


# Keep backward compatible class names
class Wide_ResNet_GAMP_SPBNN(PrunedWideResNetGAMP):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, num_models, use_gamp=True):
        # Use default pruning configuration (no pruning)
        pruning_config = {
            'pruning_rate': 0.0,
            'prune_epoch_start': 1000,  # Set to a large value to disable pruning essentially
            'final_pruning_rate': 0.0
        }
        super().__init__(depth, widen_factor, dropout_rate, num_classes, num_models, use_gamp, pruning_config)


if __name__ == '__main__':
    # Test pruned model
    pruning_config = {
        'pruning_rate': 0.3,
        'prune_epoch_start': 10,
        'final_pruning_rate': 0.5
    }

    net = PrunedWideResNetGAMP(28, 10, 0.3, 10, 4, use_gamp=True, pruning_config=pruning_config)
    net = net.cuda()

    # Test forward propagation
    test_input = torch.randn(4, 3, 32, 32).cuda()
    output = net(test_input)
    print(f"Output shape: {output.shape}")

    # Test uncertainty estimation
    output, uncertainty_info = net(test_input, return_uncertainty=True)
    print(f"Uncertainty output shape: {output.shape}")
    print(f"Uncertainty keys: {uncertainty_info.keys()}")

    # Test pruning statistics
    output, pruning_stats = net(test_input, return_pruning_stats=True)
    print(f"Pruning stats: {pruning_stats}")