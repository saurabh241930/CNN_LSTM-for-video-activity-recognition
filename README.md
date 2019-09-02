# CNN-LSTM-for-Activity-Recognition

**This method is use to detect particular activity in certain span of video**

This works by passing spatial-temporal features of video frames which is extracted  by state of the art Inception network and passing into **LSTM** network

![](https://i.imgur.com/6zQdCkV.jpg)

## Setup

*Directory Map*

```
root@526e99924bc2:/tf/CNN_LSTM# tree -d
CNN-LSTM/
├── __pycache__
├── data
│   ├── checkpoints
│   ├── logs
│   │   └── lstm
│   ├── sequences
│   ├── test
│   │   ├── ApplyEyeMakeup
│   │   ├── ApplyLipstick
│   │   ├── Archery
│   │   ├── BabyCrawling
│   ├── train
│   │   ├── ApplyEyeMakeup
│   │   ├── ApplyLipstick
│   │   ├── Archery
│   │   ├── BabyCrawling
│   └── ucfTrainTestlist
├── pipeline_v1_files
└── tests
    ├── frames
    ├── rescaled
    └── sequence


```

Install all dependcies

`pip install -r requirements.txt`

To train on UCF 101 data

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Extract data

`unrar e UCF101.rar`

Make folders inside data folder

`mkdir train && mkdir test && mkdir sequences && mkdir `

To move and extract images from video

```
python 1_move_files.py

python 2_extract_files.py

```
Extracting features with CNN model

`python extract_features.py`

Your trained model should be saved in `data/checkpoints/lstm-features.023-1.096.hdf5` it will have different name depending upon epoch and accuracy


## Now to predict on any video

Importing all required Libraries

```python
os.chdir('/CNN-LSTM')

from subprocess import call
import os
from extractor import Extractor
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from data import DataSet
import numpy as np
from natsort import realsorted, ns
import natsort
```

Importing models

```python
#Importing CNN Model 

extractor_model = Extractor()


#Importing LSTM Model 

model = load_model('data/checkpoints/lstm-features.023-1.096.hdf5')


```

Reading Video and breaking into frames

```python
# Put you Video PATH in src

src = "crawling.avi"   
dest = os.path.join("tests", "frames","%1d.jpg")

#you can adjust fps here but it may effect output

call(["ffmpeg", "-i", src,"-vf","fps=10", dest])
```

Extracting CNN features and saving into npy data

```python
#Extracting CNN features and saving into numpy array
sequence = []

i = 0
sequence_path = "tests/sequence/data_final.npy"

#change directory where your frames are located
os.chdir("tests/frames")
sequence = []

files = [f for f in os.listdir('.') if os.path.isfile(f)]

sorted_files = natsort.natsorted(files,reverse=False)

for f in sorted_files:
    i = i + 1
    features = extractor_model.extract(f)
    sequence.append(features)
    if i==40:
        break
    

np.save(sequence_path, sequence)
    
```

Loading numpy array into memory


```python
sequences = np.load("CNN-LSTM/")
# note that shape should be equal to trained sequence data
print(sequences.shape)
```

Making Prediction


```python
#Predicting

os.chdir('/tf/five-video-classification-methods')
prediction = model.predict(np.expand_dims(sequences, axis=0))

#Importing Classnames

os.chdir('CNN-LSTM/')

from data import DataSet
data = DataSet(seq_length=40, class_limit=4)


def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes


#printing classnames

data.print_class_from_prediction(np.squeeze(prediction, axis=0))

```
Output should look something like this

```
BabyCrawling: 0.99
Archery: 0.00
ApplyEyeMakeup: 0.00
ApplyLipstick: 0.00
```


