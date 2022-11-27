import socket
import time
import wave
import numpy as np
import soundfile as sf
import liblo
import datetime

# This function gets the extracted sample points from specified .wav file as well as the number of 2-second chunks of these sample points
def get_samples(filename):
	# get data from .wav file
	data, samplerate = sf.read(filename)

	seconds_2 = len(data) // (16000 * 2) # get the number of 2-second chunks based on sample rate, which is 44100 by default
	return data, seconds_2

# This function transmits the sample points to the pd receiver.
def transmit(data, number_of_chunks):

	# This variable specifies the amount of waiting time between transmissions of each chunk. It should be slightly different depending on the platform. Please scale it.
	pause = 1.9

	target = liblo.Address(5555) # Set the port
	for j in range(number_of_chunks):
		now = datetime.datetime.now() # Get timing information for scaling
		print(j, now, j * (16000 * 2))
		for i in range(160 * 2):
			# set osc message
			msg = liblo.Message("/message")
			for	k in range(100):
				msg.add(data[j * (16000 * 2) + i * 100 + k])
			liblo.send(target, msg) # Send osc message
		time.sleep(pause) # Wait for a while

# Usage
# data, n = py2pd.get_samples("../wav/sample3.wav")
# py2pd.transmit(data, n)
