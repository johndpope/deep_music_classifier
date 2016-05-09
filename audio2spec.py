###############################################################################
#
# audio2spec.py
#
###############################################################################

from lib.decode import audio2wav

def all_au_to_wav():
    in_directory = "genres/"
    out_directory = "genres_wav/"
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

if __name__ == "__main__":
    all_au_to_wav()

