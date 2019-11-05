# Speech-Emotion-Recognition-On-MELD-Dataset
This repository is for speech emotion recognition for MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation

## Results:
| Model  | Val Accuracy |
| ------------- | ------------- |
| 2D CNN & 1D CNN ensemble  | 33.13253012048193 |
|  2D CNN & 1D CNN with GRUs | 21.80722891566265 |
| Bi-LSTM|39.277108433734945|
|Bi-GRU| 40.0|



## Features:
I have used LibROSA to get Mel-frequency cepstral coefficients(MFCCs) of each audio file. 

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


