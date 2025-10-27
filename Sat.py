import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import resample
import math

# Optional plotting (comment out if not needed)
try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False


def generate_bpsk(bits, sps):
    """Return complex BPSK waveform for given bits and samples-per-symbol (sps)."""
    symbols = 2 * bits - 1  # map {0,1} -> {-1,+1}
    # pulse shape: simple rectangular (could use root-raised-cosine)
    return np.repeat(symbols.astype(np.float64), sps).astype(np.complex128)


def add_carrier(baseband, fs, carrier_offset_hz, phase=0.0):
    """Mix baseband to a carrier offset (complex exponential)"""
    t = np.arange(len(baseband)) / fs
    return baseband * np.exp(1j * (2 * np.pi * carrier_offset_hz * t + phase))


def apply_time_offset(sig, samples_offset):
    """Shift signal in time by integer samples (pad with zeros)"""
    if samples_offset > 0:
        return np.concatenate((np.zeros(samples_offset, dtype=sig.dtype), sig))[:len(sig)]
    elif samples_offset < 0:
        return np.concatenate((sig[-samples_offset:], np.zeros(-samples_offset, dtype=sig.dtype)))
    else:
        return sig


def add_awgn(sig, snr_db):
    """Add complex AWGN to achieve desired SNR (signal power / noise power)."""
    sig_power = np.mean(np.abs(sig) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise


def estimate_frequency_offset(sig, fs, search_bw=None):
    """
    Estimate coarse carrier frequency offset using FFT-based peak method.
    Returns frequency estimate in Hz.
    """
    N = len(sig)
    # compute periodogram (power spectrum)
    S = fftshift(fft(sig * np.hanning(N)))
    freqs = np.linspace(-fs/2, fs/2, N, endpoint=False)
    power = np.abs(S) ** 2
    peak_idx = np.argmax(power)
    return freqs[peak_idx]


def correct_frequency(sig, fs, freq_hz):
    """Mix signal down by estimated freq_hz (remove carrier offset)."""
    t = np.arange(len(sig)) / fs
    return sig * np.exp(-1j * 2 * np.pi * freq_hz * t)


def estimate_time_offset(ref, other):
    """
    Estimate integer-sample time offset of `other` relative to `ref` using cross-correlation.
    Positive return means `other` starts later (delayed) relative to ref.
    """
    # use FFT-based cross-correlation for speed
    n = len(ref)
    # zero-pad to 2n for circular correlation equivalence
    R = np.fft.ifft(np.fft.fft(ref, 2*n) * np.conj(np.fft.fft(other, 2*n)))
    corr = np.real(R)
    lag = np.argmax(corr)  # lag index
    # convert to signed lag in samples: lag in [0..2n-1] -> signed [-n..n-1]
    signed_lag = lag
    if signed_lag >= n:
        signed_lag -= 2*n
    return signed_lag


def normalize_amplitude(sig):
    """Normalize amplitude to unit average power (useful before coherent combining)."""
    power = np.mean(np.abs(sig)**2)
    if power == 0:
        return sig
    return sig / np.sqrt(power)


def coherent_combine(signals):
    """
    Combine list of signals coherently (assuming they are frequency-/time-/phase-aligned).
    Simple average (sum then divide) yields SNR improvement if phases align.
    """
    s_sum = np.sum(signals, axis=0)
    return s_sum / len(signals)


def harmonize_signals(raw_signals, fs):
    """
    Input: list of complex baseband arrays (same nominal length)
    Steps:
      1) Estimate & correct coarse frequency offset for each signal
      2) Choose a reference (first signal) and align others in time (integer samples)
      3) Normalize amplitude
      4) Optionally refine phase alignment (use pilot symbols or cross-correlation on known preamble)
      5) Coherently combine
    Returns: combined_signal, corrected_signals_list
    """
    L = len(raw_signals)
    # 1) Frequency estimate & correction
    freq_estimates = []
    freq_corrected = []
    for s in raw_signals:
        f_est = estimate_frequency_offset(s, fs)
        freq_estimates.append(f_est)
        freq_corrected.append(correct_frequency(s, fs, f_est))
    # 2) Time alignment (integer-sample)
    ref = freq_corrected[0]
    aligned = [ref]
    for k in range(1, L):
        other = freq_corrected[k]
        lag = estimate_time_offset(ref, other)
        # shift other by lag (positive means other is delayed -> shift left by lag)
        aligned_other = apply_time_offset(other, int(lag))
        aligned.append(aligned_other)
    # truncate to common length
    min_len = min(len(x) for x in aligned)
    aligned = [x[:min_len] for x in aligned]
    # 3) Amplitude normalize
    normed = [normalize_amplitude(x) for x in aligned]
    # 4) (Optional) Phase adjustment: we can align phases via average phase of preamble;
    # here we do a coarse per-signal phase rotation to match reference mean phase
    ref_phase = np.angle(np.mean(ref[:min(2048, min_len)]))
    phase_aligned = []
    for x in normed:
        p = np.angle(np.mean(x[:min(2048, min_len)]))
        phase_shift = ref_phase - p
        phase_aligned.append(x * np.exp(1j * phase_shift))
    # 5) Coherent combine
    combined = coherent_combine(phase_aligned)
    return combined, phase_aligned, freq_estimates


def demo_simulation():
    """Run a demo: simulate 3 signals differing by Doppler, time offset, amplitude, noise."""
    fs = 48e3  # sampling rate (Hz) for baseband simulation (arbitrary safe value)
    sps = 8    # samples per symbol
    nsymbols = 4096
    total_len = nsymbols * sps

    # create BPSK payload with a short preamble (for alignment)
    preamble_bits = np.random.randint(0, 2, size=128)
    payload_bits = np.random.randint(0, 2, size=nsymbols - len(preamble_bits))
    bits = np.concatenate((preamble_bits, payload_bits))
    baseband = generate_bpsk(bits, sps)

    # create multiple received versions with offset/doppler/noise
    rng = np.random.RandomState(2)
    sigs = []
    true_offsets = []
    for i, (doppler_hz, time_offset_samples, amp, snr_db) in enumerate([
        (220.5,  0,    1.0,  10.0),
        (-150.2, 30,   0.9,   8.0),
        (47.35, -12,   1.1,   6.0),
    ]):
        s = baseband.copy() * amp
        s = add_carrier(s, fs, doppler_hz, phase=rng.uniform(0, 2*np.pi))
        s = apply_time_offset(s, time_offset_samples)
        s = add_awgn(s, snr_db)
        sigs.append(s)
        true_offsets.append((doppler_hz, time_offset_samples))
    # Ensure equal length by truncation to max common length
    minlen = min(len(s) for s in sigs)
    sigs = [s[:minlen] for s in sigs]

    # Harmonize
    combined, corrected_list, freq_estimates = harmonize_signals(sigs, fs)

    # Compute simple SNR estimates before/after (approx)
    def approx_snr(sig, ref_signal):
        # estimate noise by subtracting reference (assuming ref_signal has similar content)
        err = sig - ref_signal
        p_sig = np.mean(np.abs(ref_signal)**2)
        p_err = np.mean(np.abs(err)**2)
        return 10*np.log10(p_sig / p_err) if p_err > 0 else np.inf

    # Use baseband (clean) as reference - truncated to minlen
    clean_ref = baseband[:minlen]
    snrs_in = [approx_snr(sigs[i][:minlen], clean_ref) for i in range(len(sigs))]
    snr_combined = approx_snr(combined, clean_ref)

    print("True Doppler/time offsets (Hz, samples):", true_offsets)
    print("Estimated carrier offsets (Hz):", freq_estimates)
    print("Input approx SNRs (dB):", ["{:.2f}".format(x) for x in snrs_in])
    print("Combined approx SNR (dB): {:.2f}".format(snr_combined))

    # Optional plots to visualize constellations / waveforms
    if HAVE_PLOT:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 2, 1)
        plt.title("Constellation: raw signals")
        for i, s in enumerate(sigs):
            plt.scatter(np.real(s[1000:1200]), np.imag(s[1000:1200]), s=1, label=f"rx{i}")
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.title("Constellation: combined")
        plt.scatter(np.real(combined[1000:1200]), np.imag(combined[1000:1200]), s=1)
        plt.subplot(3, 1, 3)
        plt.title("Waveform real part")
        t = np.arange(800) / fs
        plt.plot(t, np.real(sigs[0][:800]), label='rx0')
        plt.plot(t, np.real(combined[:800]), label='combined', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "true_offsets": true_offsets,
        "freq_estimates": freq_estimates,
        "snrs_in_db": snrs_in,
        "snr_combined_db": snr_combined,
        "combined_signal": combined
    }


if __name__ == "__main__":
    demo_simulation()
