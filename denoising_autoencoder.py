import time
import os
import json
import numpy as np
import random

from skimage.draw import polygon
from tqdm import tqdm
import imutil
from logutil import TimeSeries

latent_size = 2
batch_size = 32
iters = 50 * 1000

env = None
prev_states = None


# First, construct a dataset
def random_example():
    x = np.random.randint(15, 25)
    y = np.random.randint(15, 25)
    radius = np.random.randint(6, 12)
    rotation = np.random.uniform(0, 2*np.pi)
    noisy_input = build_box(x, y, radius, rotation)
    target_output = build_box(x=20, y=20, radius=9, rotation=rotation)
    return noisy_input, target_output


def build_box(x, y, radius, rotation):
    state = np.zeros((1, 40, 40))
    def polar_to_euc(r, theta):
        return (y + r * np.sin(theta), x + r * np.cos(theta))
    points = [polar_to_euc(radius, rotation + t) for t in [
        np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]]
    r, c = zip(*points)
    rr, cc = polygon(r, c)
    state[:, rr, cc] = 1.
    return state


def build_dataset(size=50000):
    dataset = []
    for i in tqdm(range(size), desc='Building Dataset...'):
        x, y = random_example()
        dataset.append((x, y))
    return dataset  # list of examples

dataset = build_dataset()


def get_batch(size=32):
    idx = np.random.randint(len(dataset) - size)
    examples = dataset[idx:idx + size]
    return zip(*examples)


# Now, let's build an autoencoder in Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.fc1 = nn.Linear(20*20*3, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, latent_size)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = x.view(-1, 20*20*3)

        x = self.fc1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        z = self.fc3(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # 128 x 5 x 5
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 10 x 10
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64 x 20 x 20
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64 x 40 x 40
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.cuda()

    def forward(self, z):
        x = self.fc1(z)
        x = F.leaky_relu(x, 0.2)
        x = self.bn0(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn2(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn3(x)

        x = self.deconv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn4(x)

        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x


def main():
    # Create a 40x40 monochrome image autoencoder
    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size)
    opt_encoder = optim.Adam(encoder.parameters())
    opt_decoder = optim.Adam(decoder.parameters())

    demo_batch, demo_targets = get_batch()
    vid = imutil.VideoLoop('autoencoder_reconstruction')
    ts = TimeSeries('Training', iters)

    # Train the network on the denoising autoencoder task
    for i in range(iters):
        encoder.train()
        decoder.train()

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()

        batch_input, batch_target = get_batch()

        x = torch.Tensor(batch_input).cuda()
        y = torch.Tensor(batch_target).cuda()

        z = encoder(x)
        x_hat = decoder(z)
        loss = F.binary_cross_entropy(x_hat, y)
        ts.collect('Reconstruction loss', loss)

        loss.backward()
        opt_encoder.step()
        opt_decoder.step()

        encoder.eval()
        decoder.eval()

        if i % 25 == 0:
            filename = 'iter_{:06}_reconstruction.jpg'.format(i)
            x = torch.Tensor(demo_batch).cuda()
            z = encoder(x)
            x_hat = decoder(z)
            img = torch.cat([x[:4], x_hat[:4]])
            caption = 'iter {}: orig. vs reconstruction'.format(i)
            imutil.show(img, filename=filename, resize_to=(256,512), img_padding=10, caption=caption, font_size=8)
            vid.write_frame(img, resize_to=(256,512), img_padding=10, caption=caption, font_size=12)
        ts.print_every(2)

    # Now examine the representation that the network has learned
    EVAL_FRAMES = 240
    z = torch.Tensor((1, latent_size)).cuda()
    ts_eval = TimeSeries('Evaluation', EVAL_FRAMES)
    vid_eval = imutil.VideoLoop('latent_space_traversal')
    for i in range(EVAL_FRAMES):
        theta = 2*np.pi*(i / EVAL_FRAMES)
        box = build_box(20, 20, 10, theta)
        z = encoder(torch.Tensor(box).unsqueeze(0).cuda())[0]
        ts_eval.collect('Latent Dim 1', z[0])
        ts_eval.collect('Latent Dim 2', z[1])
        caption = "Theta={:.2f} Z_0={:.3f} Z_1={:.3f}".format(theta, z[0], z[1])
        pixels = imutil.show(box, resize_to=(512,512), caption=caption, font_size=12, return_pixels=True)
        vid_eval.write_frame(pixels)
    print(ts)


if __name__ == '__main__':
    main()
