# -*- coding: cp1252 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 16/05/2020
#LABORATORY 1: SEÑALES ANALOGAS Y DIGITALES (REDES DE COMPUTADORES)

# IMPORTS
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

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

# Señal mono 1 solo canal
# 16 bit samples
# ancho de muestra 2

def is_valid_audio_file(filename):
    if not path.exists(filename):
        return False
    elif filename[-4:] != ".wav":
        return False
    else:
        return True

def plot_audio_signal(fs, wave):
    time = get_time_audio_signal(fs, wave)
    plt.figure(num="Wave in Time Domain - "+filename[:-4], figsize=(8, 5))
    plt.plot(time, wave, color="blue")
    plt.xlim(time[0], time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Wave in Time Domain')

def get_time_audio_signal(fs, wave):
    return np.arange(wave.size)/float(fs)

def plot_fft_audio_signal(fs, wave):
    fft_freq, fft_wave = get_fft_audio_signal(fs, wave)
    
    plt.figure(num="FFT in Frequency Domain - "+filename[:-4], figsize=(8, 5))
    plt.subplot(2,1,1)
    plt.plot(fft_freq, fft_wave.real, label="Real part")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('F(w)')
    plt.legend(loc=1)
    plt.title("FFT in Frequency Domain")

    plt.subplot(2,1,2)
    plt.plot(fft_freq, fft_wave.imag,label="Imaginary part")
    plt.legend(loc=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('F(w)')
    plt.xlabel("frequency (Hz)")
    plt.subplots_adjust(hspace=0.5)
    #wm = plt.get_current_fig_manager()
    #wm.window.state('zoomed')
    plt.show()

def get_fft_audio_signal(fs, wave):
    fft_freq = np.fft.fftfreq(wave.size, 1/fs)
    fft_wave = np.fft.fft(wave)
    return fft_freq, fft_wave
# MAIN
def main():
    global filename
    filename = "handel.wav"
    #read_wav_file_py(filename)
    #filename = input("Choose an audio file (.wav) to read: ")
    if is_valid_audio_file(filename):
        fs, wave = wavfile.read(filename)
        plot_audio_signal(fs, wave)
        plot_fft_audio_signal(fs, wave)
        
    else:
        print("Audio file entered is not supported or doesn't exist.")
        return 0
    #wav_file_data = read_wav_file(filename)
    #sample_rate = wav_file_data[0]
    #sample_data = wav_file_data[1]
    
    #archivo='audiofinal.wav'
    #wavfile.write(archivo, int(sample_rate), sample_data)

    # Salida a gráfico
    #plt.plot(sample_data)
    #plt.show()

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
#3. Follow Style Guidelines
#The PEP8 holds some great community-generated proposals. PEP stands for Python Enhancement Proposals- these are guidelines and standards for development. There are other PEPs like the PEP20, but PEP8 is the Python community bible for you if you want to properly style your code. This is to make sure all Python code looks and feels the same. One such guideline is to name classes with the CapWords convention.

#Use proper naming conventions for variables, functions, methods, and more.
#Variables, functions, methods, packages, modules: this_is_a_variable (snake_case)
#Classes and exceptions: CapWords (Upper Camel Case. A naming convention, also known as PascalCase)
#Protected methods and internal functions: _single_leading_underscore (snake_case)
#Private methods: __double_leading_underscore
#Constants: CAPS_WITH_UNDERSCORES

# camelCase
# PascalCase
# snake_case
# kebab-case


#Use 4 spaces for indentation. For more conventions, refer to PEP8