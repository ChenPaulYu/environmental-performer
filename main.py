import os
import json
import time
import liblo
from algo import algo
from py2pd import py2pd
from pd2py import pd2py


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    server = liblo.Server(5000)
    server.add_method(None, None, pd2py.callback, algo.algo)
    
    while True:
        s = server.recv(5000)
        if s == True:
            print("receiving...")
            while s:
                s = server.recv(1000)

        if os.path.exists("./wav/trans.wav"):
            
            instr_index = load_json('instr.json')['instr_index']
            print('instr_index: ', instr_index)
            start = time.time()
            algo.algo('./wav/trans.wav', './wav/trans.wav', int(instr_index))
            end = time.time()
            print('transfer time: ', end-start)
            
            print("transmitting...")
            # Transmit processed audio to pd.
            data, n = py2pd.get_samples("./wav/trans.wav")
            py2pd.transmit(data, n)
            os.remove("./wav/trans.wav")
            pd2py.reset()
            print("done!")
            
        print("idle...")
