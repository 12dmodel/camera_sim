import numpy as np
import torch.nn as nn

from .normalizer import *
from .n3net import N3Net
from data_generation.data_utils import cuda_like


class ModelWrapper(nn.Module):
    # This model takes care of input normalization so that outside everything
    # can be in the range of 0-1.
    def __init__(self,
                 model,
                 normalization_mode,
                 max_patch_size=None,
                 patch_buffer=None,
                 denormalize=True):
        """

        max_patch_size should be (width, height).
        """
        super().__init__()
        # This should have been renamed denoiser_model, but it will cause
        # compatibility with previously trained checkpoints.
        self.model = model

        if normalization_mode == 'constant':
            print("Using Constant Normalization")
            self.normalizer = ConstantNormalizer()
            self.denormalizer = ConstantDenormalizer()
        elif normalization_mode == 'mean':
            print("Using Mean Normalization")
            self.normalizer = MeanNormalizer()
            self.denormalizer = MeanDenormalizer()
        elif normalization_mode is None:
            print("Using No Normalization")
            self.normalizer = IdentityNormalizer()
            self.denormalizer = IdentityDenormalizer()
        else:
            raise ValueError("Invalid normalization_mode received: {}" \
                    .format(normalization_mode))
        self.max_patch_size = max_patch_size
        self.patch_buffer = patch_buffer
        self.denormalize = denormalize

    @staticmethod
    def _calc_crop_idx(idx, max_idx, patch_sz, overlap):
        start_in = idx * (patch_sz - overlap)
        if start_in >= max_idx:
            # we are out of bound
            return None
        end_in = start_in + patch_sz
        pb = int(overlap / 2)
        if idx == 0:
            start_out = 0
        else:
            start_out = start_in + pb
        end_out = end_in - pb
        if end_in >= max_idx:
            end_in = max_idx
            start_in = end_in - patch_sz
            end_out = max_idx
            start_out = end_out - patch_sz + overlap
        start_rel = start_out - start_in
        end_rel = end_out - start_in
        return start_in, end_in, start_out, end_out, start_rel, end_rel

    def forward(self, image, extra_args={}):
        networks = [self.model]

        def _process_image(img, networks, extra_args):
            output = img
            for i in range(len(networks)):
                output = networks[i](output, **extra_args)
            return output

        image, data = self.normalizer(image)
        if self.max_patch_size is None:
            output = _process_image(image, networks, extra_args)
        else:
            # Process image patch by patch and stich them up.
            # for readability
            pb = self.patch_buffer
            overlap = pb * 2
            n_batch = image.size(0)
            height, width = image.size(-2), image.size(-1)
            n_cols = np.ceil((width  - overlap) / (self.max_patch_size[0] - overlap)) + 1
            n_rows = np.ceil((height - overlap) / (self.max_patch_size[1] - overlap)) + 1
            n_rows, n_cols = int(n_rows), int(n_cols)
            # output = torch.empty_like(image)
            output = torch.zeros_like(image)
            count = torch.zeros((n_batch, 1, height, width))
            # create tapering weight
            x = torch.linspace(-1, 1, self.max_patch_size[0])
            y = torch.linspace(-1, 1, self.max_patch_size[1])
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            y = y.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            weight = 2.0 - torch.abs(x + y) - torch.abs(y - x)
            # 4 --> 2 from averaging and 2 from patch_size mapping to -1 and 1.
            saturation = overlap * 4 / (self.max_patch_size[0] + self.max_patch_size[1])
            # the factor of 2 comes from the fact that weight maxes at 2, and
            # prev statement is ration of overlap to patch_size
            saturation = saturation * 2
            weight = torch.clamp(weight, 0.01*saturation, saturation)
            count = cuda_like(count, output)
            weight = cuda_like(weight, output)
            for i in range(n_cols):
                for j in range(n_rows):
                    xs = self._calc_crop_idx(i, width,  self.max_patch_size[0], overlap)
                    ys = self._calc_crop_idx(j, height, self.max_patch_size[1], overlap)
                    # check if indices are out of bound
                    if xs is None or ys is None:
                        continue
                    x_ext, y_ext = xs[1] - xs[0], ys[1] - ys[0]
                    weight_ = weight[..., :y_ext, :x_ext]
                    patch = image[..., ys[0]:ys[1], xs[0]:xs[1]]
                    patch = _process_image(patch, networks, extra_args).detach()
                    output[..., ys[0]:ys[1], xs[0]:xs[1]] += patch * weight_
                    count[...,  ys[0]:ys[1], xs[0]:xs[1]] += weight_
            output /= count
        if self.denormalize:
            output = self.denormalizer(output, data)
        return output


def _build_submodel(arch, conf):
    if arch == "n3net":
        model = N3Net(**conf)
    else:
        raise ValueError("Architecture not recognized: {}" \
                .format(arch))
    return model


def get_model(denoising_arch_config,
              max_patch_size=None,
              patch_buffer=None):
    conf_name = denoising_arch_config["arch"] + "_config"
    model = _build_submodel(denoising_arch_config["arch"], denoising_arch_config[conf_name])
    normalization_mode = denoising_arch_config["normalization_mode"]
    return ModelWrapper(model,
                        normalization_mode,
                        max_patch_size,
                        patch_buffer)

