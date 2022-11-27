# Environmental Performer

> This repository contains the code of environmental perfromer project which is the midterm project of **Sound Design and Art Creation Course in NTU.** 

## Overview 

In our daily lives, natural sounds (e.g., birds, water, wind) are all around us, but we often don't pay attention to them. Instead, we always plug in the AirPods and listen to music on our phones. Instead, we wear earphones and listen to music most of the time. We raise a question about whether all the natural sounds are a kind of music we are unaware of. Thus, we envision a future that people can understand this kind of music performed by nature in a dimension we never knew. To listen to this kind of music, we get help from current AI technology to transfer the sound from natural sound to the music performed by the instrument. More specifically, we use the [neural waveshaping synthesis](https://github.com/ben-hayes/neural-waveshaping-synthesis) model to do the timbre transfer between natural sound and music performed by musical instruments. 


## Setup 

### Prerequisites
- python=3.8.13

### Install 
1. install liblo (only test in MacOS, if you are windows user, please refer to [link](https://liblo.sourceforge.net/README.html))
```    
brew install liblo
```

2. install nerual waveshaping synthesis

```
git clone https://github.com/ben-hayes/neural-waveshaping-synthesis.git
cd neural-waveshaping-synthesis
pip install -r requirements.txt
pip install -e .
cd ..
```

3. install other package 

```
pip install -r requirements.txt
```

### Run 

In our system, we use python as our backend to run the algorithm and use pure data as interface to interact with user. Thus, you need to first run the python backend and open the pure data to run our system. 

- run python backend
```
python main.py
```
- open pure data 
```
open main.pd
```

### Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=D4N2EQWvRNA" target="_blank">
 <img src="https://imgur.com/a/T6FI4XA" alt="Watch the video" width="640" height="480" border="10" />
</a>


