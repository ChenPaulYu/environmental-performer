import numpy as np
import json
import scipy.io.wavfile as wavf

data_t = np.array([])


def out_json(data, out_path):
    with open(out_path, 'w') as outfile:
        json.dump(data, outfile)

def reset():
	global data_t
	data_t = np.array([])
	return

def callback(path, args, types, src, params):
	global data_t
	algo = params[0]
	# algo goes here, you can do something before concate it to previous data
	# args = algo(args)
	
	out_json({'instr_index': args[-1]}, 'instr.json')
	# print(args[-1])
	args = args[:-1]
	data_t = np.concatenate([data_t, args])
	if len(data_t) % 32000 == 0: # write to wave file every 44100 datas, less effency!
		wavf.write(f"./wav/trans.wav", 16000, data_t)
		# to see how many seconds we have received
		print(f"{len(data_t) // 16000} seconds wave file trans.wav has benn written")

