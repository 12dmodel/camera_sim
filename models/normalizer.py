import torch
import torch.nn as nn


class ConstantNormalizer(nn.Module):
    # Convert to -1 to 1 range.
    def forward(self, x):
        # This has bayer mask
        has_bayer_mask = x.size(1) > 3
        if has_bayer_mask:
            mask = x[:, 3:, ...]
            x = x[:, :3, ...]
        x = (x - 0.5) * 2.0
        if has_bayer_mask:
            x = torch.cat([x, mask], dim=1)
        return x, None


class ConstantDenormalizer(nn.Module):
    # Convert back to 0 to 1 range.
    def forward(self, x, data):
        return (x * 0.5) + 0.5


class MeanNormalizer(nn.Module):
    def forward(self, x):
        # This has bayer mask
        has_bayer_mask = x.size(1) > 3
        if has_bayer_mask:
            mask = x[:, 3:, ...]
            x = x[:, :3, ...]

        x_ = x.contiguous().view(x.size(0), x.size(1), -1)
        mean = torch.mean(x_, dim=-1).unsqueeze(-1).unsqueeze(-1)
        std = torch.std(x_, dim=-1).unsqueeze(-1).unsqueeze(-1)
        x = (x - mean) / (std + 0.0001)

        if has_bayer_mask:
            x = torch.cat([x, mask], dim=1)
        return x, (mean, std)


class MeanDenormalizer(nn.Module):
    def forward(self, x, data):
        mean = data[0]
        std = data[1]
        return (x * (std + 0.0001)) + mean


class IdentityNormalizer(nn.Module):
    def forward(self, x):
        return x, None


class IdentityDenormalizer(nn.Module):
    def forward(self, x, data):
        return x
