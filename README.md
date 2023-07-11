# Event Classifier Via Audio
This program was created during working on engineering thesis entitled 
"**Identification of the existential functions of animals in the terrarium 
based on the analysis of acoustic signals**".\
For full documentation of the project please visit this page:\
[Instructable](https://www.instructables.com/Event-Classification-Via-Audio-for-Pogona-Vitticep/)

The project investigated the behavior of a bearded dragon in a terrarium.
There were distinguished three types of activities such as **running, eating
and resting (acoustic background)** of the agama. The main goal of the project was to create
a program that would recognize the animal's activity and save the information
about it to a text file along with the time and date of the activity.


The **acquisition.py** script is responsible for recording the sounds in the
terrarium at set intervals. It is possible to manipulate the recording parameters
such as recording time, recording interval, sampling rate, etc. The optimal 
parameter values were chosen by testing effectiveness of the classifier
for different values of the recording parameters. Recordings are saved to
the selected folder using the date and time as the file name.

The **audio_dataset_loader.py** script is responsible for:
1) dividing given _.wav_ file into shorter recordings with the given parameter
of "_duration_", creating appropriately named folders and saving the recordings,
2) extracting specific spectral features and MFCC (Mel-frequency cepstral coefficients)
from the given signals,
3) creating GMM models of each animal's activity based on extracted coefficients,
4) dividing prepared samples into training and test data and then training
a GMM model of each animal's activity,
5) testing the effectiveness of the classifier using different parameters such as
"_duration_", "_window_length_", "_hop_length_" etc.,
6) saving results of the training to the ._txt_ file (considering the number of correctly
classified recordings of each animal's activity),
7) drawing graphs and histograms from the samples.

The **analysis_and_classification.py** script is basically responsible for analyzing the
recording in terms of comparing the spectral features and other parameters with those in
the trained model and classifying the real time recording into appropriate group of activity. So that is:
1) loading the GMM models from a ._bin_ file,
2) extracting coefficient from the given signal and comparing them with those in trained models,
3) calculating and drawing SPL (sound pressure level) of the given signal,
4) connecting with the AGH university server via SSH.
