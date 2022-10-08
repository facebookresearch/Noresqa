# NORESQA: Speech Quality Assessment using Non-Matching References


This is a Pytorch implementation for using NORESQA. It contains minimal code to predict speech quality using NORESQA. Please see our **Neurips 2021** paper referenced below for details.

### Minimal basic usages as Speech Quality Assessment Metric.

##  Setup and basic usage

Required python libraries (latest): Pytorch with GPU support + Scipy + Numpy (>=1.14) + Librosa.
Install all dependencies in a conda environment by using:
```
conda env create -f requirements.yml
```
Activate the created environment by:
```
conda activate noresqa
```

Additional notes:
- Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the pytorch version you're using.
- Tested on Nvidia GeForce RTX 2080 GPU with Cuda (>=9.2) and CuDNN (>=7.3.0). CPU mode should also work.
- The current pretrained models support **sampling rate = 16KHz**. The provided code automatically resamples the recording to 16KHz.

Please run the metric by using:
```
usage:

python main.py --GPU_id -1 --mode file --test_file path1 --nmr path2

arguments:
--GPU_id         [-1 or 0,1,2,3,...] specify -1 for CPU, and 0,1,2,3 .. as gpu numbers
--mode           [file,list] using single nmr or a list of nmr
--test_file      [path1] -> path of the test recording
--nmr            [path2 of file, or txt file with filenames]
```

The default output of the code should look like:

```
Probaility of the test speech cleaner than the given NMR = 0.11526459
NORESQA score of the test speech with respect to the given NMR = 18.595860697038006
```

Some GPU's are non-deterministic, and so the results could vary slightly in the lsb.

Please also note that the model inherently works when the size of the input recordings are same. If they are not, then the size of the reference recording is **adjusted** to match the size of the test recording.

Please see *main.py* for more information on how to use this for your task.

### Citation

If you use this repository, please use the following to cite.

```
@inproceedings{
manocha2021noresqa,
title={{NORESQA}: A Framework for Speech Quality Assessment using Non-Matching References},
author={Pranay Manocha and Buye Xu and Anurag Kumar},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=RwASmRpLp-}
}
```

### License
The majority of NORESQA is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Librosa is licensed under the ISC license; Pytorch and Numpy are licensed under the BSD license; Scipy and Scikit-learn is licensed under the BSD-3; Libsndfile is licensed under GNU LGPL; Pyyaml is licensed under MIT License.
