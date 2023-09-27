# Top-k

This repository contains all the code to reproduce the experiments from 
'_Harnessing the Power of Choices in Decision Tree Learning_' 
by Guy Blanc, Jane Lange, Chirag Pabbaraju, Colin Sullivan, Li-Yang Tan, and Mo Tiwari, published at NeurIPS 2023.

The main code is a fork of the PyDL8.5 repository.

Below, we have instructions on how to reproduce all of our results.
If you have a question about our code, please submit a Github issue.

## Set Up The Environment

Our code only supports Python 3.9.

### M1 Macs

If you're on an M1 Mac, you'll need to install `gosdt` from source; see the instructions 
[here](https://github.com/ubc-systopia/gosdt-guesses/blob/main/doc/build.md). Afterwards, install
the other dependencies and the `topk` Python package with

```
python -m pip install -r requirements_m1mac.txt
python -m pip install .
```

### Other Platforms

If you're not on an M1 Mac, you can install all dependencies and then build the `topk` Python package directly:

```
python -m pip install -r requirements.txt
python -m pip install .
```

## Recreate Experimental Results

You can recreate all of our experimental results and plots with

```
python main.py
```