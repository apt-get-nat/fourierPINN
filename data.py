import torch
import numpy as np
from astropy.io import fits
import os

class SHARPData(torch.utils.data.Dataset):
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    def __getitem__(self, index):
        'Generates one sample of data'
        
        sharp = self.list_IDs[index]
        files = glob.glob(_rawfolder + f'hmi.sharp_cea_720s.{sharp}.*.Br.fits')
        Brfits = fits.open(files[0])
        Br = torch.Tensor(Brfits[1].data)

        files = glob.glob(_rawfolder + f'hmi.sharp_cea_720s.{sharp}.*.Bp.fits')
        Bpfits = fits.open(files[0])
        Bp = torch.Tensor(Bpfits[1].data)

        files = glob.glob(_rawfolder + f'hmi.sharp_cea_720s.{sharp}.*.Bt.fits')
        Btfits = fits.open(files[0])
        Bt = torch.Tensor(Btfits[1].data)
        
        sharpnum = torch.torch.IntTensor(sharp)
        
        return {'Br': Br, 'Bt': Bt, 'Bp': Bp, 'sharp': sharp}
    
# class SHARPDataset(torch.utils.data.Dataset):
#     def __init__(self,transform=None, pre_transform=None, pre_filter=None):
#         self.allSharps = get_allsharps()
#         super().__init__(transform, pre_transform, pre_filter)
                
#     def len(self):
#         return len(self.allSharps)
    
#     def get(self, index):
#         data = SHARPData(index)
#         return data
    
def get_allsharps():
    # return _allsharps
    sharplist = glob.glob(_rawfolder + '*.Br.fits')
    sharps = []
    for f in sharplist:
        arr = f.split('.')
        sharps.append(int(f[2]))