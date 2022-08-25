import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch import Tensor

from lib.model import TinyNeRF
from lib.utils import posenc, get_rays, render_rays


def one_epoch(height,
              width,
              focal_length,
              camera_to_world,
              near,
              far,
              n_samples,
              encoding_function,
              model,
              chunk_size) -> Tensor:
    ray_origins, ray_directions = get_rays(height,
                                           width,
                                           focal_length,
                                           camera_to_world)

    rgb_pred, _, _ = render_rays(encoding_function,
                                 model,
                                 ray_origins,
                                 ray_directions,
                                 near,
                                 far,
                                 n_samples,
                                 chunk_size)

    return rgb_pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("tiny_nerf_data.npz")

    images = data["images"]
    height, width = images.shape[1:3]

    camera_to_world = data["poses"]
    camera_to_world = torch.from_numpy(camera_to_world).to(device)

    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).to(device)

    near = 2.0
    far = 6.0

    testimg, testpose = images[101], camera_to_world[101]
    testimg = torch.from_numpy(testimg).to(device)

    images = torch.from_numpy(images[:100, ..., :3]).to(device)

    plt.imshow(testimg.detach().cpu().numpy())
    plt.show()

    l_embed = 6
    n_samples = 32
    chunk_size = 16384
    lr = 5e-3
    num_iters = 1000
    display_every = 100

    def encode(x): return posenc(x, l_embed=l_embed)

    model = TinyNeRF(D=l_embed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    psnrs = []
    iternums = []
    imgs = []

    for i in range(num_iters):
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_camera_to_world = camera_to_world[target_img_idx].to(device)

        rgb_predicted = one_epoch(height,
                                  width,
                                  focal_length,
                                  target_camera_to_world,
                                  near,
                                  far,
                                  n_samples,
                                  encode,
                                  model,
                                  chunk_size)

        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % display_every == 0:
            rgb_predicted = one_epoch(height,
                                      width,
                                      focal_length,
                                      testpose,
                                      near,
                                      far,
                                      n_samples,
                                      encode,
                                      model,
                                      chunk_size)

            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            print("Loss:", loss.item())
            psnr = -10. * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.subplot(2, 5, i // 100 + 1)
            plt.title(f"Epoch: {i}")
            plt.axis("off")
            plt.imshow(rgb_predicted.detach().cpu().numpy())

    plt.show()
    print("Done!")
