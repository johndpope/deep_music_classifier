###############################################################################
#
# audio2spec.py
#
###############################################################################

import numpy as np
from matplotlib import pyplot as plt

from lib.decode import audio2wav
from lib.spectrogram import savespec

genres = ["blues",
          "classical",
	      "country",
	      "disco",
	      "hiphop",
	      "jazz",
	      "metal",
	      "pop",
	      "reggae",
	      "rock"]

def all_au_to_wav():
    in_directory = "genres/"
    out_directory = "genres_wav/"

    for genre in genres:
        print('\nBegin directory: "' + in_directory + '/' + genre + ".")
        for i in range(100):
            if i < 10:
                j = "0" + str(i)
            else:
                j = str(i)
            infile = in_directory \
                     + genre + "/" \
                     + genre + ".000" + j + ".au"
            outfile = out_directory \
                     + genre + "/" \
                     + genre + ".000" + j + ".wav"
            audio2wav(infile, outfile)
        print('Finished directory.')

def all_wav_to_spec():
    in_directory = "genres_wav/"
    out_directory = "spectrograms/"

    for genre in genres:
        print('\nBegin directory: "' + in_directory + '/' + genre + ".")
        for i in range(100):
            if i < 10:
                j = "0" + str(i)
            else:
                j = str(i)
            infile = in_directory \
                     + genre + "/" \
                     + genre + ".000" + j + ".wav"
            outfile = out_directory \
                     + genre + "/" \
                     + genre + ".000" + j + ".spec.png"
            savespec(infile, outfile)
        print('Finished directory.')


if __name__ == "__main__":
    #all_au_to_wav()
    all_wav_to_spec()
