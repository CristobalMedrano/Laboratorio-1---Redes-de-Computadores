# -*- coding: cp1252 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 16/05/2020
#LABORATORY 1: SEÑALES ANALOGAS Y DIGITALES (REDES DE COMPUTADORES)

# IMPORTS
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import wave as wave_info


# CONSTANTS 
# GLOBAL VARIABLES
global filename
# CLASSES
# FUNCTIONS
def read_wav_file(wav_filename):
    """ Read a WAV file.
    
    Return a tuple with sample rate (Hz) and data from a WAV file

    Parameters:
    ----------
    wav_filename : string
        Input wav filename. 

    Returns:
    -------
    wav_file_data: tuple
        Tuple with sample rate and sample data from a WAV file.
   """
    sample_rate, sample_data = wavfile.read(wav_filename)
    return (sample_rate, sample_data)

def is_valid_audio_file(filename):
    """ Check if it's a valid audio file.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    filename : string
        Input audio filename. 

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
   """
    if not path.exists(filename):
        return False
    elif filename[-4:] != ".wav":
        return False
    else:
        return True

def get_time_audio_signal(fs, wave):
    """ Get time range in a wave, based on the frequency.
    
    Return the wave time range
    
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    Returns:
    -------
    ndarray: ndarray
        Array of evenly spaced values with the time range.
   """
    return np.arange(wave.size)/float(fs)

def plot_audio_signal(fs, wave, time, title):
    """ Plot the audio wave
        
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    time: ndarray
        Array of evenly spaced values with the time range.
    title: string
        Chart title
   """
    plt.figure(num=title+" - "+filename[:-4], figsize=(8, 5))
    plt.plot(time, wave, color="blue", label="f(t)")
    plt.legend(loc=1)
    plt.xlim(time[0], time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('f(t)')
    plt.title(title)

def get_fourier_transform(fs, wave):
    """ Get the fourier transform of the audio signal.
    
    Return the Discrete Fourier Transform sample frequencies and the
    truncated or zero-padded input, transformed along the axis
    
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    Returns:
    -------
    fft_freq: ndarray
        The Discrete Fourier Transform sample frequencies.
    fft_wave: complex ndarray
        The truncated or zero-padded input, transformed along the axis.
   """
    fft_freq = np.fft.fftfreq(wave.size, 1/fs)
    fft_wave = np.fft.fft(wave)
    return fft_freq, fft_wave


def get_inverse_fourier_transform(fft_freq, fft_wave):
    """ Get the inverse fourier transform of the fourier transform.
    
    Return the inverse fourier transform of a wave

    Parameters:
    ----------
    fft_freq : int
        The Discrete Fourier Transform sample frequencies. 
    fft_wave : numpy array
        The truncated or zero-padded input, transformed along the axis.
    Returns:
    -------
    ifft_wave: complex ndarray
        The truncated or zero-padded input, transformed along the axis.
   """
    ifft_wave = np.fft.ifft(fft_wave, fft_freq.size)
    return ifft_wave

def plot_fourier_transform(fft_freq, fft_wave, fs, title):
    """ Plot the Fourier Transform.
        
    Parameters:
    ----------
    fft_freq : int
        The Discrete Fourier Transform sample frequencies.
    fft_wave : numpy array
        The truncated or zero-padded input, transformed along the axis.
    fs : int
        Sample rate of wave. 
    title: string
        Chart title
   """
    plt.figure(num=title+" - "+filename[:-4], figsize=(8, 5))
    plt.plot(fft_freq, abs(fft_wave), color="blue", label="|F(w)|")
    plt.legend(loc=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|F(w)|')
    plt.xlim(-fs/2, fs/2)
    plt.title(title)

def plot_inverse_fourier_transform(fs, wave, time, title):
    """ Plot the Inverse Fourier Transform
        
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    time: ndarray
        Array of evenly spaced values with the time range.
    title: string
        Chart title
   """
    plt.figure(num=title+" - "+filename[:-4], figsize=(8, 5))
    plt.plot(time, wave, color="blue", label="ifft(t)")
    plt.legend(loc=1)
    plt.xlim(time[0], time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('ifft(t)')
    plt.title(title)

def plot_specgram(fs, wave, title):
    """ Plot the specgram
        
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    title: string
        Chart title
   """
    plt.figure(num=title+" - "+filename[:-4], figsize=(8, 5))
    plt.specgram(x=wave, Fs=fs)
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

def get_low_pass_filter(fs, wave, cut_fs):
    """ Obtain the wave filter by lowpass.
    
    Returns: The filtered output with the same shape as `wave`.
     
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave.
    cut_fs: int
        Cutoff frequency.
    Returns:
    -------
    y: ndarray
        The filtered output with the same shape as `wave`.
   """
    order = 4
    b, a = signal.butter(order, cut_fs, 'lowpass', analog=False, fs=fs)
    y = signal.filtfilt(b, a, wave, axis=0)
    return y

def get_high_pass_filter(fs, wave, cut_fs):
    """ Obtain the wave filter by highpass.
    
    Returns: The filtered output with the same shape as `wave`.
     
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave.
    cut_fs: int
        Cutoff frequency.
    Returns:
    -------
    y: ndarray
        The filtered output with the same shape as `wave`.
   """
    order = 4
    b, a = signal.butter(order, cut_fs, 'highpass', analog=False, fs=fs)
    y = signal.filtfilt(b, a, wave, axis=0)
    return y

def get_band_pass_filter(fs, wave, low_cut_fs, high_cut_fs):
    """ Obtain the wave filter by bandpass.
    
    Returns: The filtered output with the same shape as `wave`.
     
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave.
    low_cut_fs: int
        Low cutoff frequency.
    high_cut_fs: int
        High cutoff frequency.
    Returns:
    -------
    y: ndarray
        The filtered output with the same shape as `wave`.
   """
    order = 4
    b, a = signal.butter(order, [low_cut_fs, high_cut_fs], 'bandpass', analog=False, fs=fs)
    y = signal.filtfilt(b, a, wave, axis=0)
    return y

# MAIN
def main():
    """ Main function of program """
    global filename
    filename = "handel.wav"
    #filename = input("Choose an audio file (.wav) to read: ")
    if is_valid_audio_file(filename):

        # Getting the wave details
        wave_details = wave_info.open(filename)
        wave_params = wave_details.getparams()
        print(filename+" audio details:")
        print("nchannels= "+ str(wave_params[0]))
        print("sampwidth= "+ str(wave_params[1]))
        print("framerate= "+ str(wave_params[2]))
        print("nframes= "+ str(wave_params[3]))

        # Reading the audio file
        fs, wave = wavfile.read(filename)
        time = get_time_audio_signal(fs, wave)

        # Plotting the wave
        print("plotting...")
        plot_audio_signal(fs, wave, time, "Wave in Time Domain")

        # Getting Fourier Transform and Inverse Fourier Transform
        fft_freq, fft_wave = get_fourier_transform(fs, wave)
        ifft_wave = get_inverse_fourier_transform(fft_freq, fft_wave)

        # Plotting the Fourier Transform
        plot_fourier_transform(fft_freq, fft_wave, fs, 'Fourier Transform in Frequency Domain')
        
        # Plotting the Inverse Fourier Transform
        plot_inverse_fourier_transform(fs, ifft_wave.real, time, "Inverse Fourier Transform in Time Domain")
        
        # Plotting the Specgram of signal
        plot_specgram(fs, wave, "Specgram of signal")

        # Getting the IIR filters:
        low_pass_wave = get_low_pass_filter(fs, wave, 475)
        high_pass_wave = get_high_pass_filter(fs, wave, 515)
        band_pass_wave = get_band_pass_filter(fs, wave, 1, 1560)

        # Getting Fourier Transform
        fft_low_pass_freq, fft_low_pass_wave = get_fourier_transform(fs, low_pass_wave)
        fft_high_pass_freq, fft_high_pass_wave = get_fourier_transform(fs, high_pass_wave)
        fft_band_pass_freq, fft_band_pass_wave = get_fourier_transform(fs, band_pass_wave)

        # Plotting the Fourier Transform of the Filters
        plot_fourier_transform(fft_low_pass_freq, fft_low_pass_wave, fs, "(Low Pass Filter) Fourier Transform in Frequency Domain")
        plot_fourier_transform(fft_high_pass_freq, fft_high_pass_wave, fs, "(High Pass Filter) Fourier Transform in Frequency Domain")
        plot_fourier_transform(fft_band_pass_freq, fft_band_pass_wave, fs, "(Band Pass Filter) Fourier Transform in Frequency Domain")
        
        # Plotting the Specgram of filtered signals
        plot_specgram(fs, low_pass_wave, "(Low Pass Filter) Specgram of filtered signal")
        plot_specgram(fs, high_pass_wave, "(High Pass Filter) Specgram of filtered signal")
        plot_specgram(fs, band_pass_wave, "(Band Pass Filter) Specgram of filtered signal")
        
        # Getting Inverse Fourier Transform of the Filters.
        ifft_low_pass_wave = get_inverse_fourier_transform(fft_low_pass_freq, fft_low_pass_wave)
        ifft_high_pass_wave = get_inverse_fourier_transform(fft_high_pass_freq, fft_high_pass_wave)
        ifft_band_pass_wave = get_inverse_fourier_transform(fft_band_pass_freq, fft_band_pass_wave)

        # Plotting the Inverse Fourier Transform of the Filters.
        plot_inverse_fourier_transform(fs, ifft_low_pass_wave.real, time, "(Low Pass Filter) Inverse Fourier Transform in Time Domain")
        plot_inverse_fourier_transform(fs, ifft_high_pass_wave.real, time, "(High Pass Filter) Inverse Fourier Transform in Time Domain")
        plot_inverse_fourier_transform(fs, ifft_band_pass_wave.real, time, "(Band Pass Filter) Inverse Fourier Transform in Time Domain")

        # Write the filtered audios in a file
        wavfile.write(filename[:-4]+"(low_pass).wav", int(fs), np.int16(low_pass_wave))
        wavfile.write(filename[:-4]+"(high_pass).wav", int(fs), np.int16(high_pass_wave))
        wavfile.write(filename[:-4]+"(band_pass).wav", int(fs), np.int16(band_pass_wave))
        
        # Show all plots.
        plt.show()
    
    else:
        print("Audio file entered is not supported or doesn't exist.")
        return 0
    
    print("Program Finished!")
    return 0

main()

#REFERENCIAS
#http://blog.espol.edu.ec/telg1001/audio-en-formato-wav/
#https://data-flair.training/blogs/python-best-practices/
#https://realpython.com/python-pep8/
#https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
#https://www.tutorialspoint.com/scipy/scipy_fftpack.htm
#http://wwwens.aero.jussieu.fr/lefrere/master/SPE/docs-python/scipy-doc/generated/scipy.io.wavfile.read.html
#http://blog.espol.edu.ec/telg1001/fft-python-scipy/
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html#numpy.fft.fft
#https://gist.github.com/leouieda/9043213
#https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
#https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
#https://stackoverflow.com/questions/59979354/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-fftfreq
#https://www.youtube.com/watch?v=juYqcck_GfU
#https://www.programcreek.com/python/example/93228/scipy.io.wavfile.write
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
#https://www.programcreek.com/python/example/93228/scipy.io.wavfile.write
#https://dsp.stackexchange.com/questions/46509/what-do-high-and-low-order-have-a-meaning-in-fir-filter
#https://www.youtube.com/watch?v=MN0SF5n8e0Q
#https://www.minidsp.com/applications/dsp-basics/fir-vs-iir-filtering
#https://es.wikipedia.org/wiki/IIR
#https://uvirtual.usach.cl/moodle/pluginfile.php/208452/mod_resource/content/1/Laboratorio%201%20Redes%202020-1.pdf
#https://docs.python.org/3/library/wave.html
#https://numpy.org/doc/1.18/reference/generated/numpy.fft.fft.html