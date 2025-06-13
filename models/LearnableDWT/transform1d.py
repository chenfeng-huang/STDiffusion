import torch
import torch.nn as nn
import pywt
import numpy as np
from . import lowlevel

def wt_visual(h0, level=5):
    '''Get wavelet function from wavelet object.
    Params
    ------
    w_transform: obj
        DWT1d or DWT2d object
    '''
    h1 = _low_to_high(h0)

    h0 = list(h0.squeeze().detach().cpu().numpy())[::-1]
    h1 = list(h1.squeeze().detach().cpu().numpy())[::-1]

    my_filter_bank = (h0, h1, h0[::-1], h1[::-1])
    my_wavelet = pywt.Wavelet('My Wavelet', filter_bank=my_filter_bank)
    wave = my_wavelet.wavefun(level=level)
    (phi, psi, x) = wave[0], wave[1], wave[4]

    return phi, psi, x, my_wavelet

def _low_to_high(x0):
    """Converts lowpass filter to highpass filter"""
    n = x0.size(2)
    seq = (-1) ** torch.arange(n, device=x0.device)
    return torch.flip(x0, (0, 2)) * seq


class DWT1D(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, _ = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, _ = wave[0], wave[1]

        h0 = np.array(h0[::-1]).ravel()
        t = torch.get_default_dtype()
        h0 = torch.tensor(h0, dtype=t).reshape((1, 1, -1))
        # Convert h0 to Parameter here
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.J = J
        self.mode = mode

    def decompose(self, x):
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = lowlevel.mode_to_int(self.mode)

        h1 = _low_to_high(self.h0)
        #print(self.h0)
        #import ipdb; ipdb.set_trace()
        for j in range(self.J):
            x0, x1 = lowlevel.AFB1D.forward(x0, self.h0, h1, mode)
            highs.append(x1)

        return x0, highs
    
    def reconstruct(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = lowlevel.mode_to_int(self.mode)

        h1 = _low_to_high(self.h0)

        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = lowlevel.SFB1D.forward(x0, x1, self.h0, h1, mode)
        return x0



class DWT1DForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, _ = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, _ = wave[0], wave[1]

        h0 = np.array(h0[::-1]).ravel()
        t = torch.get_default_dtype()
        h0 = torch.tensor(h0, dtype=t).reshape((1, 1, -1))
        # Convert h0 to Parameter here
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.J = J
        self.mode = mode

    def forward(self, x):
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = lowlevel.mode_to_int(self.mode)

        h1 = _low_to_high(self.h0)
        #print(self.h0)
        #import ipdb; ipdb.set_trace()
        for j in range(self.J):
            x0, x1 = lowlevel.AFB1D.forward(x0, self.h0, h1, mode)
            highs.append(x1)

        return x0, highs

class DWT1DInverse(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, _ = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, _ = wave[0], wave[1]

        t = torch.get_default_dtype()
        g0 = torch.tensor(g0, dtype=t).reshape((1, 1, -1))
        self.g0 = nn.Parameter(g0, requires_grad=True)


        self.mode = mode

    def _create_g1(self, g0):
        """Converts lowpass filter to highpass filter"""
        n = g0.size(2)
        seq = (-1) ** torch.arange(n, device=g0.device)
        return torch.flip(g0, (0, 2)) * seq

    def forward(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = lowlevel.mode_to_int(self.mode)

        g1 = _low_to_high(self.g0)
        #print(self.g0)
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = lowlevel.SFB1D.forward(x0, x1, self.g0, g1, mode)
        return x0

