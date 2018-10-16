import time
import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries

latent_size = 4
batch_size = 32
iters = 10 * 1000

env = None
prev_states = None


class BoxEnv():
    def __init__(self):
        # The "true" latent space is four values: x, y, height, width
        self.x = np.random.randint(10, 30)
        self.y = np.random.randint(10, 30)
        self.radius = np.random.randint(2, 6)
        self.rotation = np.random.randint(0, np.pi)
        self.build_state()

    def build_state(self):
        from skimage.draw import polygon
        self.state = np.zeros((40,40,3))
        points = [
            (self.y - self.radius, self.x - self.radius),
            (self.y - self.radius, self.x + self.radius),
            (self.y + self.radius, self.x + self.radius),
            (self.y + self.radius, self.x - self.radius),
        ]
        r, c = zip(*points)
        rr, cc = polygon(r, c)
        self.state[rr, cc] = 1.


def build_dataset(size=50000):
    dataset = []
    for i in tqdm(range(size)):
        env = BoxEnv()
        example = np.moveaxis(env.state, -1, 0)
        dataset.append(example)
    return dataset  # list of examples


dataset = build_dataset()


def get_batch(size=32):
    idx = np.random.randint(len(dataset) - size)
    examples = dataset[idx:idx + size]
    return examples


# ok now we build a neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # 1x40x40
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 32x40x40
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64x20x20
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64x10x10
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64x5x5
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        # 64x3x3
        self.fc1 = nn.Linear(64*3*3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, latent_size)
        self.cuda()

    def forward(self, x):
        # Input: batch x 32 x 32
        #x = x.unsqueeze(1)
        # batch x 1 x 32 x 32
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 64*3*3)
        x = self.fc1(x)
        x = self.bn_fc1(x)

        z = self.fc2(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512)
        self.bn0 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # 128 x 5 x 5
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 10 x 10
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64 x 20 x 20
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64 x 40 x 40
        self.deconv5 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
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
    # ok now, can the network learn the task?
    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size)
    opt_encoder = optim.Adam(encoder.parameters(), lr=0.0001)
    opt_decoder = optim.Adam(decoder.parameters(), lr=0.0001)

    ts = TimeSeries('Training', iters)

    for i in range(iters):
        encoder.train()
        decoder.train()

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()

        examples = get_batch()

        x = torch.Tensor(examples).cuda()
        z = encoder(x)
        x_hat = decoder(z)

        loss = F.binary_cross_entropy(x_hat, x)
        ts.collect('Reconstruction loss', loss)

        loss.backward()
        opt_encoder.step()
        opt_decoder.step()

        encoder.eval()
        decoder.eval()

        if i % 100 == 0:
            filename = 'iter_{:06}_reconstruction.jpg'.format(i)
            img = torch.cat([x[:2], x_hat[:2]])
            caption = 'iter {}: orig. vs reconstruction'.format(i)
            imutil.show(img, filename=filename, resize_to=(256,256), img_padding=10, caption=caption, font_size=8)
        ts.print_every(2)
    print(ts)


if __name__ == '__main__':
    main()
