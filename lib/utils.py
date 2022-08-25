from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def posenc(x: Tensor, l_embed: int = 6) -> Tensor:
    rets = [x]

    frequency_bands = 2.0 ** torch.linspace(0.0,
                                            l_embed - 1,
                                            l_embed,
                                            dtype=x.dtype,
                                            device=x.device,
                                            )

    for i in frequency_bands:
        for func in [torch.sin, torch.cos]:
            rets.append(func(x * i))

    return torch.cat(rets, dim=-1)


def get_rays(
        height: int,
        width: int,
        focal_length: float,
        camera_to_world: Tensor):
    ii, jj = torch.meshgrid(torch.arange(width).to(camera_to_world),
                            torch.arange(height).to(camera_to_world),
                            indexing="ij")
    ii, jj = ii.transpose(-1, -2), jj.transpose(-1, -2)

    directions = torch.stack([(ii - width * .5) / focal_length,
                              -(jj - height * .5) / focal_length,
                              -torch.ones_like(ii)],
                             dim=-1)

    ray_directions = torch.sum(
        directions[..., None, :] * camera_to_world[:3, :3], dim=-1)
    ray_origins = camera_to_world[:3, -1].expand(ray_directions.shape)

    return ray_origins, ray_directions


def render_rays(embed_fn,
                network_fn,
                rays_o: Tensor,
                rays_d: Tensor,
                near: float,
                far: float,
                n_samples: int,
                chunk_size: int = 16384,
                rand: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    z_vals = torch.linspace(near, far, n_samples).to(rays_o)

    noise_shape = list(rays_o.shape[:-1]) + [n_samples]

    if rand:
        z_vals = z_vals + \
            torch.rand(noise_shape).to(rays_o) * (far - near) / n_samples

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts_flat = pts.reshape((-1, 3))
    pts_flat = embed_fn(pts_flat)

    batches = batchify(pts_flat, chunk_size=chunk_size)

    predictions = []
    for batch in batches:
        predictions.append(network_fn(batch))

    raw = torch.cat(predictions, dim=0)

    raw = torch.reshape(raw, list(pts.shape[:-1]) + [4])

    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    dists = torch.cat((z_vals[...,
                              1:] - z_vals[...,
                                           :-1],
                       torch.tensor([1e10],
                                    dtype=rays_o.dtype,
                                    device=rays_o.device).expand(z_vals[...,
                                                                        :1].shape)),
                      dim=-1)

    alpha = 1.0 - torch.exp(-sigma_a * dists)

    cumm_product = torch.cumprod(1.0 - alpha + 1e-10, -1)
    cumm_product = torch.roll(cumm_product, 1, -1)
    cumm_product[..., 0] = 1.0

    weights = alpha * cumm_product

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


def batchify(inputs: Tensor, chunk_size: int = 1024 * 8):
    return [inputs[i: i + chunk_size]
            for i in range(0, inputs.shape[0], chunk_size)]
