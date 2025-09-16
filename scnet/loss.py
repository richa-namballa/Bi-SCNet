import torch
import torch.nn as nn
import torch.nn.functional as F


def spec_rmse_loss(estimate, sources, stft_config):
    """
    Computes the RMSE loss between real and imaginary components of
    a spectrogram separately for musical source separation.

    Parameters
    ----------
    estimate : torch.Tensor
        Estimated signals of shape (batch, nsrc, channels, samples)
    sources : torch.Tensor
        Reference signals of shape (batch, nsrc, channels, samples)
    stft_config : dict
        Configuration of hyperparameters for computing the STFT

    Returns
    -------
    float
        Scalar loss value.
    """
    # (batch, nsrc, channels, samples)
    _, _, _, lenc = estimate.shape

    # collapse to only the time dimension (batch * nsrc * channels, samples)
    spec_estimate = estimate.view(-1, lenc)
    spec_sources = sources.view(-1, lenc)

    # compute complex-valued stft (flattened_batch, freq_bins, frames)
    # stft hyperparameters stored in stft_config
    spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)

    # separate the real and imaginary parts of complex numbers
    # (flattened_batch, freq_bins, frames, 2) where 2 represents (real, imag)
    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    # restore original dimensions
    # (batch, nrc, channels, freq_bins, frames, 2)
    new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
    spec_estimate = spec_estimate.view(*new_shape)
    spec_sources = spec_sources.view(*new_shape)

    # compute the element-wise squared error
    # (r_ref - r_est)^2 and (i_ref - i_est)^2
    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')

    # average over (channels, freq_bins, frames, 2)
    # apply square root (RMSE)
    # average over sources and batch
    # output scalar value
    dims = tuple(range(2, loss.dim()))
    loss = loss.mean(dims).sqrt().mean(dim=(0, 1))  

    return loss


def gcc_phat_cc(x1, x2, tau_max, eps=1e-6):
    """
    Computes the cross-correlation coefficients between two
    signals (left and right channels) using the GCC-PHAT
    algorithm.

    Parameters
    ----------
    x1 : torch.Tensor
        Left channel (batch, nsrc, samples)
    x2 : torch.Tensor
        Right channel (batch, nsrc, samples)
    tau_max : int
        Maximum lag value in samples
    eps : float
        Epsilon to avoid zero division (default: 1e-6)

    Returns
    -------
    torch.Tensor
        Cross correlation coefficients
        (batch, nsrc, 2T + 1)
    """
    # (batch, nsrc, samples)
    B, S, T = x1.shape

    # zero-pad to account for negative lag
    N = 2 * T

    # compute FFT along time dimension (real-valued)
    X1 = torch.fft.rfft(x1, n=N, dim=-1)
    X2 = torch.fft.rfft(x2, n=N, dim=-1)
    numerator = X1 * torch.conj(X2)
    R = numerator / (torch.abs(numerator) + eps)  # avoid zero division

    cc = torch.fft.irfft(R, n=N)  # (batch, nsrc, 1, N)
    cc = torch.fft.fftshift(cc, dim=-1)  # shift so 0 sample lag is in middle

    # restrict search window if there is a maximum delay
    middle = cc.shape[-1] // 2
    if tau_max is not None:
        cc = cc[..., (middle-tau_max):(middle+tau_max+1)]

    return cc


def itd_loss(pred, target, tau_max):
    """
    Computes the ITD loss as described by Hernandez-Olivan, et al. (2024).

    Parameters
    ----------
    pred : torch.Tensor
        Estimated signals of shape (batch, nsrc, channels, time)
    target : torch.Tensor
        Reference signals of shape (batch, nsrc, channels, time)
    tau_max : int
        Maximum lag value in samples

    Returns
    -------
    float
        Scalar loss value.
    """
    # check for two channels
    assert pred.shape[2] == 2 and target.shape[2] == 2
    
    # compute cross correlation using GCC-PHAT between left and
    # right channels of predicted and estimated signals
    pred_cc = gcc_phat_cc(pred[..., 0,:], pred[..., 1,:], tau_max)
    target_cc = gcc_phat_cc(target[..., 0,:], target[..., 1,:], tau_max)

    # compute mse
    squared_error = (pred_cc - target_cc) ** 2
    loss = squared_error.mean(dim=-1)

    return loss.mean()



class BinauralLoss(nn.Module):
    """
    Multi-task loss meant for binaural audio signal separation which
    penalizes the distortion of interaural cues.

    Parameters
    ----------
    stft_config : dict
        Configuration of hyperparameters for computing the STFT
    sr : int
        Sample rate
    max_lag : float
        Maximum lag / delay to allow in cross-correlation computation
        Default value is 1ms (0.001s)
    rmse_weight : float
        Weight of signal-level loss in loss summation
    itd_weight : float
        Weight of spatial loss in loss summation
    """
    def __init__(self, stft_config, sr=44100, max_lag=1e-3, rmse_weight=1, itd_weight=0.1):
        super(BinauralLoss, self).__init__()

        self.stft_config = stft_config
        self.tau_max = int(max_lag * sr)
        self.rmse_weight = rmse_weight
        self.itd_weight = itd_weight

        # store separate loss values for logging
        self.signal_loss = 1e6
        self.spatial_loss = 1e6

    def forward(self, estimates, targets):
        """
        Multi-task loss meant for binaural audio signal separation which
        penalizes the distortion of interaural cues.

        Parameters
        ----------
        estimates : torch.Tensor
            Estimated signals of shape (batch, nsrc, channels, samples)
        targets : torch.Tensor
            Reference signals of shape (batch, nsrc, channels, samples)
        """

        loss = 0

        if self.rmse_weight > 0:
            self.signal_loss = spec_rmse_loss(estimates, targets, self.stft_config)
            loss += self.rmse_weight * self.signal_loss

        if self.itd_weight > 0:
            self.spatial_loss = itd_loss(estimates, targets, self.tau_max)
            loss += self.itd_weight * self.spatial_loss

        return loss
