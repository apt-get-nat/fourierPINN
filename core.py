import torch
import numpy as np

class toReal(torch.nn.Module):
    def __init__(self):
        super(toReal, self).__init__()
    def forward(self, x):
        return torch.cat((x.real,x.imag))
class toComplex(torch.nn.Module):
    def __init__(self):
        super(toComplex, self).__init__()
    def forward(self, x):
        return x.cfloat()

class kernelNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity):
        super(kernelNN, self).__init__()
        
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = torch.nn.ModuleList()
        
        self.layers.append(toReal())
        layers[0] = 2*layers[0]

        for j in range(self.n_layers):

            if j != self.n_layers - 1:
                self.layers.append(torch.nn.Linear(layers[j], layers[j+1]))
                self.layers.append(nonlinearity())
            else:
                self.layers.append(toComplex())
                self.layers.append(torch.nn.Linear(layers[j], layers[j+1], dtype=torch.cfloat))

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class FourierNN(torch.nn.Module):
    def __init__(self, layers, signalN = 32):
        super(FourierNN, self).__init__()
        self.signalN = signalN
        self.kernel = kernelNN([3*2*signalN**2]+layers+[3*4*signalN**3],torch.nn.Sigmoid)
    def forward(self, x):
        s = self.signalN
        
        xn = x['Br'].shape[0]
        yn = x['Br'].shape[1]
        zn = np.max([xn,yn])
        
        bdx = torch.fft.rfft2(x['Bp'])
        bdy = torch.fft.rfft2(-x['Bt'])
        bdz = torch.fft.rfft2(x['Br'])
        bdx_pos = bdx[0:s,0:s]
        bdx_neg = bdx[-s:,0:s]
        bdy_pos = bdy[0:s,0:s]
        bdy_neg = bdy[-s:,0:s]
        bdz_pos = bdz[0:s,0:s]
        bdz_neg = bdz[-s:,0:s]
        
        invec = torch.cat((torch.flatten(bdx_pos),torch.flatten(bdx_neg),
                           torch.flatten(bdy_pos),torch.flatten(bdy_neg),
                           torch.flatten(bdz_pos),torch.flatten(bdz_neg)))
        
        outvec = self.kernel.forward(invec)
        
        fullx = torch.zeros(xn,yn,zn,dtype=torch.cfloat)
        fully = torch.zeros(xn,yn,zn,dtype=torch.cfloat)
        fullz = torch.zeros(xn,yn,zn,dtype=torch.cfloat)
        
        fullx[0:s,0:s,0:s] = torch.reshape(outvec[0:s**3],[s,s,s])
        fullx[0:s,-s:,0:s] = torch.reshape(outvec[1*s**3:2*s**3],[s,s,s])
        fullx[-s:,0:s,0:s] = torch.reshape(outvec[2*s**3:3*s**3],[s,s,s])
        fullx[-s:,-s:,0:s] = torch.reshape(outvec[3*s**3:4*s**3],[s,s,s])
        
        fully[0:s,0:s,0:s] = torch.reshape(outvec[4*s**3:5*s**3],[s,s,s])
        fully[0:s,-s:,0:s] = torch.reshape(outvec[5*s**3:6*s**3],[s,s,s])
        fully[-s:,0:s,0:s] = torch.reshape(outvec[6*s**3:7*s**3],[s,s,s])
        fully[-s:,-s:,0:s] = torch.reshape(outvec[7*s**3:8*s**3],[s,s,s])
        
        fullz[0:s,0:s,0:s] = torch.reshape(outvec[8*s**3:9*s**3],[s,s,s])
        fullz[0:s,-s:,0:s] = torch.reshape(outvec[9*s**3:10*s**3],[s,s,s])
        fullz[-s:,0:s,0:s] = torch.reshape(outvec[10*s**3:11*s**3],[s,s,s])
        fullz[-s:,-s:,0:s] = torch.reshape(outvec[11*s**3:12*s**3],[s,s,s])
        
        Bx = torch.fft.irfftn(fullx)
        By = torch.fft.irfftn(fully)
        Bz = torch.fft.irfftn(fullz)
        
        # compute derivatives
        xmodes = torch.arange(0,int(xn/2+1))
        xmodes = torch.hstack((xmodes[0:-1],-xmodes.flip(0)))
        xmodes = xmodes[0:-1]
        ymodes = torch.arange(0,int(yn/2+1))
        ymodes = torch.hstack((ymodes[0:-1],-ymodes.flip(0)))
        ymodes = ymodes[0:-1]
        zmodes = torch.arange(0,int(zn))
        kx,ky,kz = torch.meshgrid(xmodes,ymodes,zmodes,indexing='ij')
        
        dxBx = torch.fft.irfftn(torch.mul(fullx,1j*kx))
        dyBx = torch.fft.irfftn(torch.mul(fullx,1j*ky))
        dzBx = torch.fft.irfftn(torch.mul(fullx,1j*kz))
        dxBy = torch.fft.irfftn(torch.mul(fully,1j*kx))
        dyBy = torch.fft.irfftn(torch.mul(fully,1j*ky))
        dzBy = torch.fft.irfftn(torch.mul(fully,1j*kz))
        dxBz = torch.fft.irfftn(torch.mul(fullz,1j*kx))
        dyBz = torch.fft.irfftn(torch.mul(fullz,1j*ky))
        dzBz = torch.fft.irfftn(torch.mul(fullz,1j*kz))
        
        return {  'Bx':  Bx,   'By':  By,   'Bz':  Bz,
                'dxBx':dxBx, 'dyBx':dyBx, 'dzBx':dzBx,
                'dxBy':dxBy, 'dyBy':dyBy, 'dzBy':dzBy,
                'dxBz':dxBz, 'dyBz':dyBz, 'dzBz':dzBz
               }
        
        