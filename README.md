# NORESQA: Speech Quality Assessment using Non-Matching References


This is a Pytorch implementation for using NORESQA. The NORESQA framework uses non-matching
references (NMT) along with the given test speech signal to estimate speech quality. Under this framework
we have two metrics:

- *NORESQA-score*: A metric based on SI-SDR for speech. The model predicts absolute relative SI-SDR [1] between test and NMR and probability of test cleaner than the NMR
- *NORESQA-MOS*: NORESQA-MOS is designed to estimate Mean Opinion Score (MOS) [2]. The output of NORESQA-MOS is MOS score.

The detailed of NORESQA framework and *NORESQA-score* is in our **Neurips 2021** paper referenced below.The details of *NORESQA-MOS* is in our **Interspeech 2022** paper.

### Using this library as Speech Quality Assessment Metric.

##  Setup and basic usage

Required python libraries (latest): Pytorch with GPU support + Scipy + Numpy (>=1.14) + Librosa + fairseq.

Install all dependencies in a conda environment by using:
```
conda env create -f requirements.yml
```
Activate the created environment by:
```
conda activate noresqa
```
Set CONFIG_PATH in main.py. This is path of *Wav2Vec 2.0 Base* model used for instantiating the NORESQA-MOS model. For default **download *Wav2Vec 2.0 Base* model from [this link](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) and put inside models/ directory.**

The models in models/ directory have been uploaded using ```git lfs```. You may want to clone using ```git lfs``` to clone properly. 

Additional notes:
- Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the pytorch version you're using.

- *NORESQA-MOS* predictions are based on assumptions that the provided NMR(s) is(are) clean. Use only clean NMRs to predict MOS.

- The current pretrained models support **sampling rate = 16KHz**. The provided code automatically resamples the recording to 16KHz.

Please run the metric by using:
```
usage:

python main.py --GPU_id -1 --metric_type 1 --mode file --test_file path1 --nmr path2

arguments:
--GPU_id         [-1 or 0,1,2,3,...] specify -1 for CPU, and 0,1,2,3 .. as gpu numbers
--metric_type    0 --> *NORESQA-score*, 1 --> *NORESQA-MOS*
--mode           [file,list] using single nmr or a list of nmr
--test_file      [path1] -> path of the test recording
--nmr            [path2 of file, or txt file with filenames]
```

For *NORESQA-score* the output should look like:

```
Probaility of the test speech cleaner than the given NMR = 0.11526459
NORESQA score of the test speech with respect to the given NMR = 18.595860697038006
```

For *NORESQA-MOS* the output should look like:
```
MOS score of the test speech (assuming NMR is clean) = 2.003323554992676
```
Note that for *NORESQA-MOS*, the model's default output is relative MOS. The actual MOS output is ```5-(model_output)```.


Some GPU's are non-deterministic, and so the results could vary slightly in the lsb.

Please also note that the model inherently works when the size of the input recordings are same. If they are not, then the size of the reference recording is **adjusted** to match the size of the test recording.

Please see *main.py* for more information on how to use this for your task.

### Citation

If you use this repository, please use the following to cite.

```
@inproceedings{
noresqa,
title={{NORESQA}: A Framework for Speech Quality Assessment using Non-Matching References},
author={Pranay Manocha and Buye Xu and Anurag Kumar},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://proceedings.neurips.cc/paper/2021/file/bc6d753857fe3dd4275dff707dedf329-Paper.pdf}
}

@inproceedings{
noresqamos,
title={Speech Quality Assessment through MOS using Non-Matching References},
author={Pranay Manocha and Anurag Kumar},
booktitle={Interspeech},
year={2022},
url={https://arxiv.org/abs/2206.12285}
}
```

### License
The majority of NORESQA is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Librosa is licensed under the ISC license; Pytorch and Numpy are licensed under the BSD license; Scipy and Scikit-learn is licensed under the BSD-3; Libsndfile is licensed under GNU LGPL; Pyyaml is licensed under MIT License.

### References
[1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 626-630). IEEE.

[2] Loizou, Philipos C. "Speech quality assessment." In Multimedia analysis, processing and communications, pp. 623-654. Springer, Berlin, Heidelberg, 2011.
