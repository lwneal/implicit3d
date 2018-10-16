# Augmented Autoencoders

An implementation of the proof-of-concept experiment from
*Implicit 3D Orientation Learning for 6D Object Detection from RGB Images* by Martin Sundermeyer et al. ECCV 2018.

## Install and Run

```bash
git clone https://github.com/lwneal/implicit3d
cd implicit3d
pip install -r requirements.txt
python denoising_autoencoder.py
```

Position and scale are treated as noise factors, and the autoencoder learns to be invariant to them, learning only the desired factor (rotation).

## Learned Representation

![Rotation representation](https://github.com/lwneal/implicit3d/raw/master/representation.jpg)
Graph showing the value of the encoding of an image as the rotation of the image changes.
Note the period of the graph- due to rotational symmetry, the same representation is repeated with four offsets.

## Reconstructions

Top 4: Original images. Bottom 4: Reconstructed images after training.
![Reconstructions](https://github.com/lwneal/implicit3d/raw/master/reconstructions.jpg)
