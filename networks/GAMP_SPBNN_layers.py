import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class PrunedGAMPModule(nn.Module):
    """GAMP module with pruning incorporated"""

    def __init__(self, input_dim, hidden_dim, num_iterations=3, pruning_rate=0.2):
        super(PrunedGAMPModule, self).__init__()
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pruning_rate = pruning_rate

        # GAMP parameters
        self.theta = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.phi = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        # Pruning related parameters
        self.theta_mask = nn.Parameter(torch.ones_like(self.theta), requires_grad=False)
        self.phi_mask = nn.Parameter(torch.ones_like(self.phi), requires_grad=False)
        self.theta_importance = nn.Parameter(torch.zeros_like(self.theta), requires_grad=False)
        self.phi_importance = nn.Parameter(torch.zeros_like(self.phi), requires_grad=False)

        # Pruning control
        self.apply_pruning = False
        self.pruning_warmup = 50  # Pruning warmup epoch count

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.theta, mean=0.0, std=0.1)
        nn.init.normal_(self.phi, mean=0.0, std=0.1)


    def get_effective_weights(self):
        """Get effective weight matrices to ensure dimension consistency"""
        if self.apply_pruning:
            # Apply mask
            theta = self.theta * self.theta_mask
            phi = self.phi * self.phi_mask

            # Check and fix dimension issues
            # Ensure the number of columns of theta (input_dim) matches the number of rows of phi (input_dim)
            # Ensure the number of rows of theta (hidden_dim) matches the number of columns of phi (hidden_dim)

            # Check if theta has completely pruned rows
            theta_row_sums = torch.sum(self.theta_mask, dim=1)
            zero_theta_rows = (theta_row_sums == 0)

            # Check if phi has completely pruned columns
            phi_col_sums = torch.sum(self.phi_mask, dim=0)
            zero_phi_cols = (phi_col_sums == 0)

            # If there are completely pruned rows or columns, restore at least one connection
            if torch.any(zero_theta_rows):
                for i in range(self.hidden_dim):
                    if zero_theta_rows[i]:
                        # Restore the first element of this row
                        self.theta_mask.data[i, 0] = 1.0
                        theta = self.theta * self.theta_mask

            if torch.any(zero_phi_cols):
                for i in range(self.hidden_dim):
                    if zero_phi_cols[i]:
                        # Restore the first element of this column
                        self.phi_mask.data[0, i] = 1.0
                        phi = self.phi * self.phi_mask
        else:
            theta = self.theta
            phi = self.phi

        return theta, phi

    def forward(self, x, prior_mean, prior_var):
        batch_size = x.size(0)

        # Get effective weight matrices
        theta, phi = self.get_effective_weights()

        # Debug info: check dimensions
        if batch_size != x.size(0):
            print(f"Warning: batch_size mismatch: {batch_size} vs {x.size(0)}")

        # Initialization - ensure correct dimensions
        v = torch.ones(batch_size, theta.size(0), device=x.device) * 0.01  # Use actual row count of theta
        r = torch.zeros(batch_size, theta.size(0), device=x.device)  # Use actual row count of theta

        # Ensure matrix dimension matching
        if r.size(1) != theta.size(0):
            print(f"Error: r and theta dimension mismatch: {r.size()} vs {theta.size()}")
            # Adjust r dimension to match theta
            if r.size(1) < theta.size(0):
                # Pad with zeros if r dimension is smaller than theta
                padding = torch.zeros(batch_size, theta.size(0) - r.size(1), device=r.device)
                r = torch.cat([r, padding], dim=1)
            else:
                # Truncate if r dimension is larger than theta
                r = r[:, :theta.size(0)]



        # Damping factor
        damping = 0.8

        for iter in range(self.num_iterations):
            # Forward pass - ensure matrix multiplication dimension matching
            try:
                z = torch.matmul(r, theta.t())
            except RuntimeError as e:
                print(f"Matrix multiplication error in forward pass:")
                print(f"r shape: {r.shape}, theta shape: {theta.shape}")
                print(f"theta.t() shape: {theta.t().shape}")
                raise e

            z_var = torch.matmul(v, (theta ** 2).t()) + 1e-6

            # Output node update
            g = (x - z) / z_var
            g_var = 1 / z_var

            # Backward pass
            try:
                s = torch.matmul(g, phi)
            except RuntimeError as e:
                print(f"Matrix multiplication error in backward pass:")
                print(f"g shape: {g.shape}, phi shape: {phi.shape}")
                raise e

            s_var = torch.matmul(g_var, (phi ** 2)) + 1e-6

            # Hidden node update
            new_post_mean = (prior_mean / (prior_var + 1e-6) + s / s_var) / (1 / (prior_var + 1e-6) + 1 / s_var)
            new_post_var = 1 / (1 / (prior_var + 1e-6) + 1 / s_var)

            # Apply damping
            r = damping * r + (1 - damping) * new_post_mean
            v = damping * v + (1 - damping) * new_post_var

        return r, v


class EnsembleModule(nn.Module):
    """creates base class for BNN, in order to enable specific behavior"""

    def init(self):
        super().__init__()


class Ensemble_BatchNorm2d(nn.Module):
    def __init__(self, num_features, num_models=1, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True):
        super(Ensemble_BatchNorm2d, self).__init__()
        self.num_models = num_models
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_features, affine=affine)
             for i in range(num_models)])

    def forward(self, input):
        inputs = torch.chunk(input, self.num_models, dim=0)
        res = torch.cat(
            [l(inputs[i]) for i, l in enumerate(self.batch_norms)], dim=0)
        return res


class PrunedEnsemble_FC_GAMP(EnsembleModule):
    def __init__(self, in_features, out_features, first_layer,
                 num_models, bias=True, use_gamp=True, hidden_size=64, pruning_rate=0.3):
        super(PrunedEnsemble_FC_GAMP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.pruning_rate = pruning_rate

        # Base fully connected layer
        self.fc = nn.Linear(in_features, out_features, bias=False)

        # Pruning related parameters
        self.weight_mask = nn.Parameter(torch.ones_like(self.fc.weight), requires_grad=False)
        self.weight_importance = nn.Parameter(torch.zeros_like(self.fc.weight), requires_grad=False)
        self.apply_pruning = False
        self.pruning_warmup = 50

        self.use_gamp = use_gamp
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features), requires_grad=False)
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))

        if use_gamp:
            self.gamp_module = PrunedGAMPModule(in_features, hidden_size, pruning_rate=pruning_rate)
            self.gamp_decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, in_features)
            )
        else:
            # Traditional VAE encoder
            self.encoder_fc1 = nn.Linear(in_features, hidden_size)
            self.encoder_fcmean = nn.Linear(hidden_size, hidden_size)
            self.encoder_fcvar = nn.Linear(hidden_size, hidden_size)
            self.decoder_fc1 = nn.Linear(hidden_size, in_features)

        self.loss_latent = 0
        self.num_models = num_models

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1.0, std=0.05)
        nn.init.normal_(self.gamma, mean=1.0, std=0.05)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

        # Initialize base weights
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))


    def get_effective_weight(self):
        """Get effective weight matrix"""
        if self.apply_pruning:
            # Ensure at least some connections are retained
            if torch.sum(self.weight_mask) == 0:
                # Restore the first weight if no weights are retained
                self.weight_mask.data[0, 0] = 1.0
            return self.fc.weight * self.weight_mask
        else:
            return self.fc.weight

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def gamp_forward(self, alpha_input):
        """GAMP forward propagation"""
        batch_size = alpha_input.size(0)

        prior_mean = torch.zeros(batch_size, self.hidden_size, device=alpha_input.device)
        prior_var = torch.ones(batch_size, self.hidden_size, device=alpha_input.device) * 0.1

        try:
            z_mean, z_var = self.gamp_module(alpha_input, prior_mean, prior_var)
        except RuntimeError as e:
            print(f"GAMP forward error in FC layer:")
            print(f"alpha_input shape: {alpha_input.shape}")
            print(f"prior_mean shape: {prior_mean.shape}")
            print(f"prior_var shape: {prior_var.shape}")
            raise e

        alpha_decoded = self.gamp_decoder(z_mean)

        reconstruction_loss = F.mse_loss(alpha_decoded, alpha_input)
        kl_loss = -0.5 * torch.mean(1 + torch.log(z_var + 1e-8) - z_mean.pow(2) - z_var)

        return alpha_decoded, reconstruction_loss + 0.1 * kl_loss


    def forward(self, x):
        if self.use_gamp:
            try:
                alpha_decoded, latent_loss = self.gamp_forward(self.alpha)
            except RuntimeError as e:
                print(f"Error in FC GAMP forward:")
                print(f"x shape: {x.shape}")
                print(f"self.alpha shape: {self.alpha.shape}")
                raise e
        else:
            alpha_decoded, latent_loss = self.vae_forward(self.alpha)

        self.loss_latent = latent_loss

        # Base forward propagation with pruning applied
        effective_weight = self.get_effective_weight()

        if self.training:
            curr_bs = x.size(0)
            indices = torch.randint(high=self.num_models, size=(curr_bs,), device=self.alpha.device)

            alpha = torch.index_select(alpha_decoded, 0, indices)
            gamma = torch.index_select(self.gamma, 0, indices)
            bias = torch.index_select(self.bias, 0, indices) if self.bias is not None else None

            # Use pruned weights
            result = F.linear(x * alpha, effective_weight) * gamma
            return result + bias if bias is not None else result
        else:
            if self.first_layer:
                x = torch.cat([x for i in range(self.num_models)], dim=0)

            batch_size = int(x.size(0) / self.num_models)
            alpha = torch.cat([alpha_decoded for i in range(batch_size)], dim=0)
            gamma = torch.cat([self.gamma for i in range(batch_size)], dim=0)

            if self.bias is not None:
                bias = torch.cat([self.bias for i in range(batch_size)], dim=0)
                result = F.linear(x * alpha, effective_weight) * gamma + bias
            else:
                result = F.linear(x * alpha, effective_weight) * gamma

            return result


class PrunedEnsemble_Conv2d_GAMP(EnsembleModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=100, train_gamma=True,
                 bias=True, use_gamp=True, hidden_size=64, pruning_rate=0.3):
        super(PrunedEnsemble_Conv2d_GAMP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.pruning_rate = pruning_rate

        # Base convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)

        # Pruning related parameters
        self.weight_mask = nn.Parameter(torch.ones_like(self.conv.weight), requires_grad=False)
        self.weight_importance = nn.Parameter(torch.zeros_like(self.conv.weight), requires_grad=False)
        self.apply_pruning = False
        self.pruning_warmup = 50

        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels), requires_grad=False)
        self.train_gamma = train_gamma
        self.use_gamp = use_gamp
        self.num_models = num_models

        if use_gamp:
            self.gamp_module = PrunedGAMPModule(in_channels, hidden_size, pruning_rate=pruning_rate)
            self.gamp_decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, in_channels)
            )
        else:
            self.encoder_fc1 = nn.Linear(in_channels, hidden_size)
            self.encoder_fcmean = nn.Linear(hidden_size, hidden_size)
            self.encoder_fcvar = nn.Linear(hidden_size, hidden_size)
            self.decoder_fc1 = nn.Linear(hidden_size, in_channels)

        if train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.first_layer = first_layer
        self.loss_latent = 0

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1., std=0.1)
        if self.train_gamma:
            nn.init.normal_(self.gamma, mean=1., std=0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize convolution weights
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))


    def get_effective_weight(self):
        """Get effective weight matrix"""
        if self.apply_pruning:
            # Ensure at least some connections are retained
            if torch.sum(self.weight_mask) == 0:
                # Restore the first weight if no weights are retained
                self.weight_mask.data[0, 0, 0, 0] = 1.0
            return self.conv.weight * self.weight_mask
        else:
            return self.conv.weight

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def gamp_forward(self, alpha_input):
        """GAMP forward propagation"""
        batch_size = alpha_input.size(0)
        prior_mean = torch.zeros(batch_size, self.hidden_size, device=alpha_input.device)
        prior_var = torch.ones(batch_size, self.hidden_size, device=alpha_input.device) * 0.1

        try:
            z_mean, z_var = self.gamp_module(alpha_input, prior_mean, prior_var)
        except RuntimeError as e:
            print(f"GAMP forward error in Conv layer:")
            print(f"alpha_input shape: {alpha_input.shape}")
            print(f"prior_mean shape: {prior_mean.shape}")
            print(f"prior_var shape: {prior_var.shape}")
            raise e

        alpha_decoded = self.gamp_decoder(z_mean)

        reconstruction_loss = F.mse_loss(alpha_decoded, alpha_input)
        kl_loss = -0.5 * torch.mean(1 + torch.log(z_var + 1e-8) - z_mean.pow(2) - z_var)

        return alpha_decoded, reconstruction_loss + 0.1 * kl_loss


    def forward(self, x):
        if self.use_gamp:
            try:
                alpha_decoded, latent_loss = self.gamp_forward(self.alpha)
            except RuntimeError as e:
                print(f"Error in Conv GAMP forward:")
                print(f"x shape: {x.shape}")
                print(f"self.alpha shape: {self.alpha.shape}")
                raise e
        else:
            alpha_decoded, latent_loss = self.vae_forward(self.alpha)

        self.loss_latent = latent_loss

        # Base forward propagation with pruning applied
        effective_weight = self.get_effective_weight()

        if not self.training and self.first_layer:
            x = torch.cat([x for i in range(self.num_models)], dim=0)

        num_examples_per_model = int(x.size(0) / self.num_models)
        if num_examples_per_model == 0:
            num_examples_per_model = 1

        alpha = torch.cat([alpha_decoded for _ in range(num_examples_per_model)], dim=0)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)

        if self.train_gamma:
            gamma = torch.cat([self.gamma for _ in range(num_examples_per_model)], dim=0)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        if self.bias is not None:
            bias = torch.cat([self.bias for _ in range(num_examples_per_model)], dim=0)
            bias = bias.unsqueeze(-1).unsqueeze(-1)

        # Perform convolution with pruned weights
        result = F.conv2d(x * alpha, effective_weight, stride=self.conv.stride,
                          padding=self.conv.padding, groups=self.conv.groups)

        if self.train_gamma:
            result = result * gamma

        return result + bias if self.bias is not None else result


# Keep backward compatible class names
class Ensemble_FC_GAMP(PrunedEnsemble_FC_GAMP):
    def __init__(self, in_features, out_features, first_layer,
                 num_models, bias=True, use_gamp=True, hidden_size=64):
        super().__init__(in_features, out_features, first_layer, num_models,
                         bias, use_gamp, hidden_size, pruning_rate=0.0)


class Ensemble_Conv2d_GAMP(PrunedEnsemble_Conv2d_GAMP):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=100, train_gamma=True,
                 bias=True, use_gamp=True, hidden_size=64):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         groups, first_layer, num_models, train_gamma, bias,
                         use_gamp, hidden_size, pruning_rate=0.0)