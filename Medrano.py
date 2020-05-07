# -*- coding: cp1252 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 16/05/2020
#LABORATORY 1: SEÑALES ANALOGAS Y DIGITALES (REDES DE COMPUTADORES)

# IMPORTS
from scipy.io import wavfile

# CONSTANTS 
# GLOBAL VARIABLES
# CLASSES
# FUNCTIONS
def read_wav_file(wav_filename):
    """ Read a WAV file.
    
    Return a tuple with sample rate (in samples/sec) and data from a WAV file

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

# MAIN
def main():
    wav_file_data = read_wav_file("handel.wav")
    print(wav_file_data[0])
    print(wav_file_data[1])
    return 0

main()
#REFERENCIAS
#http://blog.espol.edu.ec/telg1001/audio-en-formato-wav/
#https://data-flair.training/blogs/python-best-practices/
#https://realpython.com/python-pep8/
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