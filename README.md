# Speech-Emotion-Recognition-On-MELD-Dataset
This repository is for speech emotion recognition for MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation

## Results:
| Model  | Val Accuracy |
| ------------- | ------------- |
|2D CNN|43.253 |
| 2D CNN & 1D CNN ensemble  | 33.132|
|  2D CNN & 1D CNN with GRUs |51.927|
| Bi-LSTM| 40.0|
|Bi-GRU|38.915|



## Features:
I have used LibROSA to get Mel-frequency cepstral coefficients(MFCCs) of each audio file. The MFCCs are of the form of 2D numpy arrays. I have also calculated the mean values of the MFCC numpy arrays to get additional features to be used by 1D CNNs. 

## Dataset:
The dataset is a collection of audio files with over 8000 utterances/phrases/conversations from the TV sitcom "Friends".

|Dataset |	Disgust |	Fear |	Happy |	Neutral |	Sad |	Total |
| :--- | :-------------: |:-------------: | :-------------: | :-------------: |:-------------: |  ---: | 
|Training set| 	232 |	216 |	1609 |	4592 |	705 |	7354 |
|Validation set |	28 |	25 |	181 |	517 |	79 |	830 |

### Issues:
* The dataset suffers from sever class imbalance. It 'Neutral' class has over 4.5K data points in the training set while 'Fear' only has 216. Such an imbalance is also observed in the validation set.
* Data is noisy, with the actor's voice being flooded by the background laughter. I was unable to seperate the background laughter from the vocals, making classification extremely difficult.
* Dataset is very small for properly understanding more nuanced emotions such as 'Disgust' and 'Fear'.

### Compensatory measures to deal with dataset issues:
* Choosing a smaller training set. But this has problem of missing out on much of the available data resource. I have used a dataset with maximum limit of 1000 data points per class as my training dataset.
* Using data augmentaion to increase dataset size. Since my dataset shrunk in size while trying to compensate for data imbalance, it was neccessary to increase my dataset size using data augmentaion techniques. I have used adding noise and random shifting data augmentaion techniques. 

## Models Used:
### 2D CNN:
It is a basic model which simply takes the MFCCs and performs 2D convolutions on it, flattens it and reduces the linear layer to the number of label categories. Such structure is usually used for audio classification tasks. It performs well on the validation set, however it overwhelmingly predicts 'Neutral' class.


 <img src="https://github.com/Azithral/Speech-Emotion-Recognition-On-MELD-Dataset/blob/master/Images/2DCNN.JPG" width = 500> 
 
### 1D CNN:
It is an even simpler model than 2D CNN and relies on the mean values of MFCC numpy arrays to be its features. 

### Bi-LSTM
The Bi-LSTM model was also tested with MFCC features. 
<img src="https://github.com/Azithral/Speech-Emotion-Recognition-On-MELD-Dataset/blob/master/Images/BiLSTM.JPG" width = 500> 

### Bi-GRU
The Bi-GRU model was also tested with MFCC features.  
<img src="https://github.com/Azithral/Speech-Emotion-Recognition-On-MELD-Dataset/blob/master/Images/BiGRU.JPG" width = 500> 

# Citations:
S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W. EmotionLines: An Emotion Corpus of Multi-Party Conversations. arXiv preprint arXiv:1802.08379 (2018).



