import os
from tqdm import tqdm
import webdataset as wds
from torch.utils.data import DataLoader
from ldm.data.laion import LaionAestheticsTrain

if __name__ == '__main__':
    base_dir = '../laion-a6p/'
    tars = [
        base_dir + fn 
        for fn in os.listdir(base_dir) 
        if fn.endswith('.tar')
    ]

    len(tars)

    # ds = wds.WebDataset(tars)
    ds = LaionAestheticsTrain(
                size= 256,
            degradation= 'pil_nearest',
    )
    dl = DataLoader(ds, 32, num_workers=4)
    for i in tqdm(dl):
        pass
# exit(0)
# ---

# import torch

# sd = torch.load('../sd-v1-5/model.ckpt')
# sdx = sd['state_dict']

# out = {}
# for k in sdx.keys():
#     if k.startswith('first_stage_model'):
#         out[k[18:]] = sdx[k]

# sd.keys()

# torch.save(out, '../sd-v1-5/ae.pt')
