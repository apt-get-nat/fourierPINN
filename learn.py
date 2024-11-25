import torch
from random import gauss
import numpy as np
from tqdm.notebook import tqdm

def runEpoch(model, traindata, optmethod = torch.optim.Adam):
    loss_div_cum = 0
    loss_lox_cum = 0
    loss_loy_cum = 0
    loss_loz_cum = 0
    loss_bdx_cum = 0
    loss_bdy_cum = 0
    loss_bdz_cum = 0
    
    if model.training:
        optimizer = optmethod(model.parameters())
        
    for X in tqdm(traindata):
        Y = model.forward(X)
        loss_div = torch.sum(torch.pow(Y['dxBx'] + Y['dyBy'] + Y['dzBz'],2).ravel()) / torch.numel(Y['Bz'])
        loss_div_cum += loss_div.item()
        
        loss_lox = torch.sum(torch.pow(torch.mul(Y['dzBx']-Y['dxBz'],Y['Bz']) - 
                                        torch.mul(Y['dxBy']-Y['dyBx'],Y['By']),2).ravel()) / torch.numel(Y['Bz'])
        loss_lox_cum += loss_lox.item()
        
        loss_loy = torch.sum(torch.pow(torch.mul(Y['dxBy']-Y['dyBx'],Y['Bx']) -
                                        torch.mul(Y['dyBz']-Y['dzBy'],Y['Bz']),2).ravel()) / torch.numel(Y['Bz'])
        loss_loy_cum += loss_loy.item()
        
        loss_loz = torch.sum(torch.pow(torch.mul(Y['dyBz']-Y['dzBy'],Y['By']) -
                                        torch.mul(Y['dzBx']-Y['dxBz'],Y['Bx']),2).ravel()) / torch.numel(Y['Bz'])
        loss_loz_cum += loss_loz.item()
        
        loss_bdx = torch.sum(torch.pow(Y['Bx'][:,:,0]-X['Bp'],2).ravel()) / torch.numel(X['Br'])
        loss_bdx_cum += loss_bdx.item()
        loss_bdy = torch.sum(torch.pow(Y['By'][:,:,0]+X['Bt'],2).ravel()) / torch.numel(X['Br'])
        loss_bdy_cum += loss_bdy.item()
        loss_bdz = torch.sum(torch.pow(Y['Bz'][:,:,0]-X['Br'],2).ravel()) / torch.numel(X['Br'])
        loss_bdz_cum += loss_bdz.item()
        
        
        if model.training:
            
            optimizer.zero_grad()
#             loss_proj = _randunit(7)
            loss_proj = [1,1,1,1,1,1,1]
        
            loss  = loss_div*loss_proj[0] + loss_lox*loss_proj[1] + loss_loy*loss_proj[2] + loss_loz*loss_proj[3]
            loss += loss_bdx*loss_proj[4] + loss_bdy*loss_proj[5] + loss_bdz*loss_proj[6]
            loss.backward()
            
#             print(loss_div.item(),loss_lox.item(),loss_loy.item(),loss_loz.item(),loss_bdx.item(),loss_bdy.item(),loss_bdz.item())
            print(loss.item())
            optimizer.step()
        
    return {'div':loss_div_cum,
            'lox':loss_lox_cum, 'loy':loss_loy_cum, 'loz':loss_loz_cum,
            'bdx':loss_bdx_cum, 'bdy':loss_bdy_cum, 'bdz':loss_bdz_cum,
            'params':model.parameters()
           }


            
def _randunit(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [np.abs(x/mag) for x in vec]