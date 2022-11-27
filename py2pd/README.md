# pd2py

Double buffering method is used in these programs.

## File Description

`py2pd.py`

* Split the .wav file into chunks. Each chunk contains audio for one second.
* Send each chunk in osc format through the socket every two seconds.
* `pause` variable should be adjusted manually so that the intervals between each sent chunk are approximately equal to 2. It should be slightly different depending on the platform.

`py2pd.pd`

* Two arrays, `array3` and `array4`, are used alternatively to realize the real-time transmission.
* There is a counter switching between 0 and 1 whenever it gets a complete chunk
* When the counter is 0, it writes the chunk to `array3`; when the counter is 1, it writes the chunk to `array4`.
* When the counter is 1, it plays the audio in `array3`; when the counter is 0, it plays the audio in `array4`.

## Usage

* Open `py2pd.pd` in pure data and press the `reset` button.
* Run `main.py` using `python3 main.py`.