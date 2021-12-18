import torch

def drop_square(im_t, size=5, pos=None):
    """
    Gray out a square in given batch of images. If no position provided,
    do it in a random position per image. This completely nullifies any 
    gradients possibly passing through the image.
    """
    drop_mask = torch.ones_like(im_t)
    r = size // 2
    batch_size, ch, height, width = im_t.size()
    a, b = torch.randint(r, height-r-size%2+1, (2, batch_size))

    for i in range(batch_size):
        x = a[i] if pos is None else pos[0]
        y = b[i] if pos is None else pos[1]

        drop_mask[i, :, y-r:y+r+size%2, x-r:x+r+size%2] = 0
    return im_t * drop_mask

def near_one_like(input, rand_range=0.1):
    return 1 - torch.rand_like(input)*rand_range

def near_zero_like(input, rand_range=0.1):
    return torch.rand_like(input)*rand_range
