import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft
import matplotlib.colors as mc
import matplotlib.cm as cm


def convert_to_watts(dBm: np.ndarray) -> np.ndarray:
    return 10**(dBm/10)/1000


def convert_to_dBm(W: np.ndarray) -> np.ndarray:
    if np.any(W <= 0):
        print("Negative or zero values found in W:", min(W[W <= 0]))
        W[W <= 0] = 1e-15
    return 10*np.log10(W)+30


def lighten(color, amount=0.5):
    # Simple way to lighten a color for matplotlib
    rgb = np.array(mc.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def darken(color, amount=0.5):
    # Simple way to darken a color for matplotlib
    rgb = np.array(mc.to_rgb(color))
    return tuple(rgb * amount)


def load_csv_oscilloscope(filename, type='average'):
    """load csv file for a measurement done with the oscilloscope"""
    df = pd.read_csv(filename)
    time = df.iloc[:, 0].values  # First column as time (in seconds)
    if type == 'spectrum':
        # Second column as voltage (in volts)
        average = convert_to_dBm(df.iloc[:, 1].values)
    elif type == 'average':
        average = df.iloc[:, 1].values  # Second column as dBV
    return time, average


def load_csv_spectrum_analyser(filename):
    df = pd.read_csv(filename, skiprows=1)
    frequency = df.iloc[:, 0].values  # First column as frequency
    amplitude = df.iloc[:, 2].values  # Third column as amplitude
    return frequency, amplitude


def remove_background(laser_off, laser_on, domain='time', frequency=None):
    """subtracts background from the signal in the units of power in the given domain. Returns the signal and its average optical power. Assumes linear scale """
    # average optical power
    if domain == 'time':
        avg = np.mean(laser_on)
    else:
        PSD = laser_on/(frequency[1]-frequency[0])
        avg = np.trapz(PSD, frequency)

    signal = laser_on - laser_off
    # Plot for checking data
    # plt.plot(frequency, convert_to_dBm(signal), label='signal')
    # plt.plot(frequency, convert_to_dBm(laser_on), label='laser_on')
    # plt.plot(frequency, convert_to_dBm(laser_off), label='laser_off')
    # plt.legend()
    # plt.show()
    return signal, avg


def get_fft(freq, voltage):

    voltage = np.array(voltage)

    # Calculate time step (assuming uniform sampling)
    fs = freq[1]-freq[0]  # Sampling frequency

    # Perform FFT
    N = len(voltage)
    fft_values = fft(voltage)
    # getting the positive frequencies
    pos_freqs = freq[:N // 2 + 1]
    # this gives us the power spectral density
    # pos_fft_values = (2.0 / (fs * N)) * np.abs(fft_values[:N // 2 + 1])**2

    return np.abs(fft_values[:N // 2 + 1])


def get_ifft(frequency, intensity):
    frequency = frequency
    frequency = np.concatenate(
        (np.array(frequency), -np.array(frequency)[::-1][1:]))
    intensity = np.concatenate(
        (np.array(intensity), np.array(intensity)[::-1][1:]))
    # Perform IFFT
    N = len(intensity)
    ifft_values = np.fft.ifft(intensity)
    ifft_values = np.real(ifft_values)

    return ifft_values


def get_PSD(frequency, intensity):
    """
    Calculate the Power Spectral Density (PSD) from the frequency and intensity data.
    The PSD is calculated as the square of the intensity divided by the frequency step.
    """
    # Calculate the frequency step
    freq_step = frequency[1] - frequency[0]

    # Calculate PSD (the factor of 2 compensates for the fft)
    PSD = 2*intensity**2/(freq_step*len(frequency))/50

    return PSD


def get_RIN(PSD, avg_power):
    return convert_to_dBm(PSD / avg_power)


def calculate_integral(frequency, PSD, range_start=0, range_end=1e7):
    """
    Calculate the integral of the Power Spectral Density (PSD) over a specified frequency range.
    If range_end is None, it integrates to the end of the frequency array.
    """
    mask = (frequency >= range_start) & (frequency <= range_end)

    # Perform the integration using trapezoidal rule
    integral = np.trapz(PSD[mask], frequency[mask],
                        dx=frequency[1] - frequency[0])

    return np.sqrt(integral)


def spectrum_analyser():
    P_values = [-185, -136, -224, -185, -224, -136, -136, -185, -224, 0]
    I_values = [0, 0, 0, -3886, -3886, -3886, -39, -39, -39, 0]

    data_files = [f for f in os.listdir('spectra') if f.endswith(".csv")]
    amplitude = []

    for i, file in enumerate(data_files):
        frequency, a = load_csv_spectrum_analyser('spectra/'+file)
        if i == 10:
            plt.plot(frequency, a, label='background',
                     linewidth=0.2, color="gray")
            laser_off = a
        else:
            plt.plot(
                frequency, a, label=f'P = {P_values[i]} I = {I_values[i]}', marker='.', linewidth=0, alpha=0.25)
            amplitude.append(a)
        print(file)
    plt.xscale('log')
    plt.ylabel('Spectrum [dBm]')
    plt.xlabel('Frequency [Hz]')
    plt.show()

    # Define color maps
    I_color_map = {
        0: 'blue',
        -3886: 'red',
        -39: 'green'
    }

    P_dark_map = {
        -185: 'lighten',
        -136: 'none',
        -224: 'darken'
    }

    integrals = np.zeros(len(amplitude))

    for i in range(len(amplitude)-1):
        P = P_values[i]
        I = I_values[i]

        base_color = I_color_map.get(I, 'black')

        if P_dark_map.get(P, 'none') == 'lighten':
            color = lighten(base_color, 0.5)
        elif P_dark_map.get(P, 'none') == 'darken':
            color = darken(base_color, 0.5)
        else:
            color = base_color

        power_W = 10**(amplitude[i] / 10)
        corrected_PSD = power_W**2 / (frequency[0]-frequency[-1])
        P_avg = np.trapz(corrected_PSD, frequency)
        corrected_RIN = convert_to_dBm(corrected_PSD/P_avg)

        power_W = 10**(amplitude[-1] / 10)
        PSD = power_W**2 / (frequency[0]-frequency[-1])
        P_avg = np.trapz(PSD, frequency)
        RIN = convert_to_dBm(PSD/P_avg)
        # corrected_signal = dBV_to_vrms(amplitude[i])
        # signal = dBV_to_vrms(amplitude[-1])
        # background = dBV_to_vrms(laser_off)
        # avg_corrected = np.sum(corrected_signal**2*(frequency[1]-frequency[0]))
        # avg_no_corrected = np.sum(
        #     signal**2*(frequency[1]-frequency[0]))

        # corrected_signal -= background
        # signal -= background
        # corrected_signal = corrected_signal**2/(frequency[1]-frequency[0])
        # signal = signal**2/(frequency[1]-frequency[0])

        integrals[i] = np.trapz(corrected_PSD/P_avg,
                                frequency, dx=frequency[1]-frequency[0])
        no_correction_integral = np.trapz(PSD/P_avg,
                                          frequency, dx=frequency[1]-frequency[0])

        plt.plot(frequency, corrected_RIN, marker='.',
                 linewidth=0, label=f'P = {P} I = {I} int = {integrals[i]}', alpha=0.25, color=color)

    plt.plot(frequency, RIN, linewidth=0.1,
             label='no correction', alpha=0.8, color='black')
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # specify order of items in legend
    order = [0, 1, 2, 3, 5, 4, 7, 6, 8, 9]
    # add legend to plot
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Relative Intensity Noise [dBc/Hz]')
    plt.show()
    integrals[-1] = no_correction_integral
    print("Integrals:", np.sqrt(integrals))
    laser_off_W = convert_to_watts(laser_off)

    no_correction = amplitude.pop(-1)
    signal = convert_to_watts(no_correction)-laser_off_W
    noise = -convert_to_watts(no_correction)+laser_off_W
    signal_dBm = convert_to_dBm(signal)
    noise_dBm = convert_to_dBm(noise)

    np.nan_to_num(signal_dBm, copy=False)
    np.nan_to_num(noise_dBm, copy=False)
    no_correction = -signal_dBm + a+noise_dBm

    responsivity = 0.4
    integrals = []

    # for i, a in enumerate(amplitude):
    #     signal = convert_to_watts(a)-laser_off_W
    #     noise = -convert_to_watts(a)+laser_off_W
    #     signal_dBm = convert_to_dBm(signal)
    #     noise_dBm = convert_to_dBm(noise)

    #     np.nan_to_num(signal_dBm, copy=False)
    #     np.nan_to_num(noise_dBm, copy=False)
    #     relative_noise = -signal_dBm + a + noise_dBm
    #     integrals.append(np.trapz(relative_noise, f, dx=f[1]-f[0]))
    #     fig, ax = plt.subplots()
    #     ax.plot(f, relative_noise, f, no_correction, marker='.', linewidth=0, label=str(
    #         data_files[i])+"\np="+str(P_values[i]) + " I="+str(I_values[i])+"\nintegral="+str(integrals[-1]))
    #     ax.set_xscale('log')
    #     # ax.set_ylim(-160, 0)
    #     ax.legend()
    # plt.show()


def analyze_old_data(old_data):
    volt = []
    for i, file in enumerate(old_data):
        print(file)
        frequency, voltage = load_csv_oscilloscope(
            'spectra/'+file, type='average')
        mask = frequency >= 1e3
        plt.plot(frequency[mask], voltage[mask],
                 label=file, marker='.', linewidth=0, alpha=0.2)
        volt.append(voltage)
    frequency, background = load_csv_oscilloscope(
        'spectra/BCK_AVG1.CSV', type='average')
    frequency, no_PID = load_csv_oscilloscope(
        'spectra/SIG_AVG1.CSV', type='average')
    mask = frequency >= 1e3
    plt.plot(frequency[mask], background[mask],
             label='background', linewidth=0.25)
    plt.plot(frequency[mask], no_PID[mask],
             label='no PID', linewidth=0.25)
    plt.ylabel('Spectrum [dBV]')
    plt.xlabel('Frequency [Hz]')
    plt.xscale('log')
    plt.legend()
    plt.show()

    P_values = np.array([-2.297e-3, -3.6621e-3, -4.882e-3, -6.103e-3, -7.8125e-3, -9.755e-3, -
                         1.2396e-2, -2.4414e-3, -1.9531e-3, -1.709e-3, -1.4648e-3, -1.2207e-3, -9.7656e-4, -7.3242e-4, 2.9297e-3])

    sort_idx = np.argsort(np.abs(P_values))
    P_sorted = P_values[sort_idx]
    volt_sorted = [volt[i] for i in sort_idx]

    cmap = cm.gist_rainbow  # or cm.plasma, cm.viridis, etc.
    norm = plt.Normalize(np.abs(P_sorted).min(), np.abs(P_sorted).max())
    colors = [cmap(norm(abs(p))) for p in P_sorted]

    mask = frequency >= 1000
    background_w = get_ifft(frequency, convert_to_watts(background))

    integrals = []
    handles = []
    labels = []
    for v, p, color in zip(volt_sorted, P_sorted, colors):
        watts = convert_to_watts(v)
        watts_t = get_ifft(frequency, watts)
        signal, average = remove_background(watts_t, background_w)
        spectrum = get_fft(frequency, signal)
        # here I already have power units, check for the factor of 2 and the frequency step
        PSD = 2*spectrum / (frequency[1]-frequency[0])
        avg = np.trapz(PSD, frequency)
        RIN = get_RIN(PSD**2, avg)
        integrals.append(calculate_integral(frequency, PSD, range_start=1e3, range_end=1e6))
        h, = plt.plot(frequency[mask], RIN[mask], marker='.',
                      linewidth=0, color=color, label=f'P={p:.3e}', alpha=0.25)
        handles.append(h)
        labels.append(f'P={p:.3e}')
    n_watts = convert_to_watts(no_PID)
    n_watts_t = get_ifft(frequency, n_watts)
    n_signal, n_average = remove_background(n_watts_t, background_w)
    n_spectrum = get_fft(frequency, n_signal)
    n_PSD = 2*n_spectrum/(frequency[1]-frequency[0])
    n_avg = np.trapz(n_PSD, frequency)
    n_RIN = get_RIN(n_PSD**2, n_avg)
    integrals.append(calculate_integral(frequency, n_PSD, range_start=1e3, range_end=1e6))
    print(integrals)
    h, = plt.plot(frequency[mask], n_RIN[mask],
                  linewidth=0.1, color='black', label='no correction', alpha=0.25)
    handles.append(h)
    labels.append('no correction')
    plt.legend(handles, labels, ncols=4)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Relative Intensity Noise [dBc/Hz]')
    plt.xscale('log')
    plt.show()


def subtract_background(laser_off, laser_on, laser_corrected, frequency):
    trace_off = get_ifft(frequency, convert_to_watts(laser_off))
    trace_on = get_ifft(frequency, convert_to_watts(laser_on))
    trace_corrected = get_ifft(frequency, convert_to_watts(laser_corrected))
    # average optical power
    avg_no_corrected = np.mean(trace_on**2)
    avg_corrected = np.mean(trace_corrected**2)

    signal_trace = (trace_on - trace_off)
    corrected_signal_trace = (trace_corrected - trace_off)
    # Plot for checking data

    # perform fft
    signal = get_fft(frequency, signal_trace)
    corrected_signal = get_fft(frequency, corrected_signal_trace)
    return signal, corrected_signal, avg_no_corrected, avg_corrected

    # plt.plot(frequency, convert_to_dBm(signal), label='laser on')
    # plt.plot(frequency, convert_to_dBm(corrected_signal), label='signal')
    # plt.legend()
    # plt.show()


def get_spectrum(signal, corrected_signal, avg_no_corrected, avg_corrected, frequency):

    mask = frequency >= 1000

    RIN = convert_to_dBm(signal/avg_no_corrected)
    corrected_RIN = convert_to_dBm(corrected_signal/avg_corrected)
    integrals = np.trapz(corrected_signal[mask]/avg_corrected,
                         frequency[mask], dx=frequency[1]-frequency[0])
    no_correction_integral = np.trapz(
        signal[mask]/avg_no_corrected, frequency[mask], dx=frequency[1]-frequency[0])

    return integrals, no_correction_integral, RIN, corrected_RIN


# spectrum_analyser()
# import trace
data_files = [f for f in os.listdir('spectra') if f.endswith(".CSV")]
# split the files for the different datasets
averaged_data = [f for f in data_files if f.startswith("20")]  # dBV
single_data = [f for f in data_files if f.startswith("100")]  # V
old_data = [f for f in data_files if f.startswith("PID")]  # dBmW
analyze_old_data(old_data)
# P abd I values for 20A....CSV files
P_values = [0, -139.66, -186.46, -186.46, -139.68, -139.68, -186.46]
I_values = [0, -2242.5, -2242.5, -3885.6, -3885.6, -1682.9, -1682.9]

volt = []
for i, file in enumerate(averaged_data):
    print(file)
    frequency, voltage = load_csv_oscilloscope('spectra/'+file, type='average')
    mask = frequency >= 1e3
    if i == 0:
        plt.plot(frequency[mask], voltage[mask],
                 label='background', marker='.', linewidth=0)
        background_v = convert_to_watts(voltage)
    elif i == 1:
        plt.plot(frequency[mask], voltage[mask],
                 label='PI_off', marker='.', linewidth=0)
        volt.append(voltage)
    else:
        plt.plot(frequency[mask], voltage[mask],
                 label=f'P = {P_values[i-2]} I = {I_values[i-2]}', marker='.', linewidth=0, alpha=0.2)
        volt.append(voltage)
plt.ylabel('Spectrum [dBV]')
plt.xlabel('Frequency [Hz]')
plt.xscale('log')
plt.legend()
plt.show()


# Define color maps
I_color_map = {
    -2242.5: 'blue',
    -3885.6: 'red',
    -1682.9: 'green'
}
P_dark_map = {
    -186.46: True,
    -139.66: False,
    -139.68: False  # In case you have both -139.66 and -139.68
}
integrals = []
background_vt = get_ifft(frequency, background_v)
for i in range(len(volt)):
    P = P_values[i]
    I = I_values[i]
    base_color = I_color_map.get(I, 'black')
    color = lighten(base_color) if P_dark_map.get(P, False) else base_color
    volts = convert_to_watts(volt[i])
    volts_t = get_ifft(frequency, volts)
    signal, average = remove_background(background_vt, volts_t)
    spectrum = (get_fft(frequency, signal))
    PSD = get_PSD(frequency, spectrum)
    avg = np.trapz(PSD, frequency)
    RIN = get_RIN(PSD, avg)
    # integrals.append(calculate_integral(frequency, PSD/avg, range_start=1e3))
    # signal, corrected_signal, avg_no_corrected, avg_corrected = subtract_background(
    #     volt[1], volt[0], volt[i], frequency)
    # integrals[i], no_correction_integral, RIN, corrected_RIN = get_spectrum(
    #     signal, corrected_signal, avg_no_corrected, avg_corrected, frequency)
    if i == 0:
        n_RIN = RIN
        n_integral = calculate_integral(frequency, PSD/avg, range_start=1e3, range_end=1e6)
    else:
        integrals.append(calculate_integral(
            frequency, PSD/avg, range_start=1e3, range_end=1e6))
        plt.plot(frequency[mask], RIN[mask], marker='.',
                 linewidth=0, label=f'P = {P} I = {I}', alpha=0.25, color=color)

plt.plot(frequency[mask], n_RIN[mask], linewidth=0.1,
         color='black', label='no correction', alpha=0.25)
# get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# specify order of items in legend
order = [0, 1, 2, 3, 5, 4, 6]
# add legend to plot
plt.legend([handles[idx] for idx in order], [labels[idx]
           for idx in order], ncols=3)
print(np.sqrt(n_integral))
print("Integrals:", np.sqrt(integrals))
# plt.legend()
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Relative Intensity Noise [dBc/Hz]')
plt.show()
assert (False)

data_files = [f for f in os.listdir('spectra') if f.endswith(".CSV")]
amplitude = []
new_f, _ = load_csv_oscilloscope('spectra/'+str(data_files[0]))
for file in data_files:
    old_f, a = load_csv_oscilloscope('spectra/'+file)
    print(file)
    amplitude.append(a)

new_a = amplitude[:8]
old_a = amplitude[8:]


# frequency1, average1 = load_csv_new('spectra/'+str(data_files[-1]), type='average')   #PID
# frequency2, average2 = load_csv_new('spectra/SIG_AVG1.CSV', type='average')   #signal
# frequency3, average3 = load_csv_new('spectra/BCK_AVG1.CSV', type='average')   #covered

mask = new_f >= 1e3
background = amplitude.pop(0)
no_correction = amplitude.pop(0)

background_W = convert_to_watts(background)
no_correction_W = convert_to_watts(no_correction)

signal = no_correction_W - background_W
over_Hz = 10*np.log10(new_f)
signal_dBm = convert_to_dBm(signal) - over_Hz

plt.plot(new_f[mask], (signal_dBm/np.mean(signal))
         [mask], marker='.', linewidth=0, label='signal')
plt.xscale('log')
plt.show()
# Convert to watts
signal = 10**(average2/10)
covered = 10**(average3/10)


# calculate noise Power
P_without_PID = signal - covered
# plt.loglog(frequency2[mask], (average2/frequency2)[mask]+10, marker = '.', linewidth = 0, label = 'signal + 10 dBm')
# plt.loglog(frequency1[mask], (average1/frequency2)[mask]+10, marker = '.', linewidth = 0, label = 'no signal + 10 dBm')
# plt.loglog(frequency3[mask], (average3/frequency2)[mask]+10, marker = '.', linewidth = 0, label = 'covered + 10 dBm')
# plt.show()

# convert back to dBm
over_Hz = 10*np.log10(frequency2)
noise_without_PID = 10*np.log10(P_without_PID)-over_Hz

# comparing the plots
for i in range(len(data_files)-2):

    frequency, average = load_csv_oscilloscope(
        'spectra/'+str(data_files[i+1]), type='average')
    # calculations
    PID = 10**(average/10)
    P_with_PID = PID - covered
    noise_with_PID = 10*np.log10(P_with_PID)
    relative_noise = noise_with_PID - over_Hz
    np.nan_to_num(relative_noise, copy=False, nan=-120)
    integrals.append(
        np.trapz(relative_noise[mask], frequency[mask], dx=frequency[1]-frequency[0]))

    fig, ax = plt.subplots()
    ax.plot(frequency[mask], relative_noise[mask], marker='.',
            linewidth=0, label=str(data_files[i+1])+"\np="+str(P_values[i]))
    ax.plot(frequency2[mask], noise_without_PID[mask],
            marker='.', linewidth=0, alpha=0.5, label='not corrected')
    ax.set_xscale('log')
    ax.legend()
plt.show()

# Plot integrated values vs P_values
plt.figure()
plt.plot(P_values, integrals, marker='o', linewidth=0)
plt.xlabel('P_values')
plt.ylabel('Integrated Relative Noise')
plt.title('Integrated Relative Noise vs P_values')
plt.grid(True)
plt.show()

plt.title('Noise Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('NSD (dBm)')
plt.grid(True)
plt.legend()
plt.show()
assert (False)
# in dBc
dBc_with_PID = noise_with_PID-average1
dBc_without_PID = noise_without_PID-average2

plt.plot(frequency2[mask], noise_with_PID[mask]-over_Hz[mask], frequency2[mask],
         noise_without_PID[mask]-over_Hz[mask], marker='.', linewidth=0)
plt.xscale('log')
# plt.figure(figsize=(10, 6))
# plt.loglog(pos_freqs[mask], pos_fft_values[mask])
plt.title('Relative intensity noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('RIN [dBc]')
plt.grid(True)
plt.legend(['PID stabilization', 'no stabilization'])
plt.show()
