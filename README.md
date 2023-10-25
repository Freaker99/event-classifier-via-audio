# Event Classifier Via Audio

**Overview**

This program was developed as part of an engineering thesis titled "Identification of the Existential Functions of Animals in the Terrarium Based on the Analysis of Acoustic Signals." For full documentation of the project, please visit this page: [Instructable](https://www.instructables.com/Event-Classification-Via-Audio-for-Pogona-Vitticep/)

## Project Description

The project focused on studying the behavior of a bearded dragon :dragon: in a terrarium. Three primary types of activities were identified, including running, eating, and resting (acoustic background) of the agama. The main objective was to create a program capable of recognizing the animal's activity and saving information, including the time and date of the activity, to a text file.

## Data Acquisition

### Acquisition.py

The `acquisition.py` script is responsible for recording sounds in the terrarium at specified intervals. It allows for the manipulation of recording parameters such as recording time, recording interval, sampling rate, and more. Optimal parameter values were determined through testing the classifier's effectiveness with varying recording parameters. Recordings are saved to the selected folder using the date and time as the file name.

## Audio Data Handling

### Audio_dataset_loader.py

The `audio_dataset_loader.py` script is responsible for several tasks, including:

- Dividing a given .wav file into shorter recordings with a specified "duration" parameter.
- Creating appropriately named folders and saving the recordings.
- Extracting specific spectral features and Mel-frequency cepstral coefficients (MFCC) from the recorded signals.
- Creating Gaussian Mixture Model (GMM) models for each animal's activity based on the extracted coefficients.
- Splitting prepared samples into training and test data.
- Training a GMM model for each animal's activity.
- Testing the classifier's effectiveness using different parameters, such as "duration," "window_length," "hop_length," etc.
- Saving the training results to a .txt file, which includes the number of correctly classified recordings for each animal's activity.
- Generating graphs and histograms from the collected samples.

## Real-time Audio Classification

### Analysis_and_classification.py

The `analysis_and_classification.py` script is primarily responsible for analyzing real-time recordings. It does so by comparing spectral features and other parameters with those in the trained models and classifying the recordings into the appropriate activity group. The tasks include:

- Loading the GMM models from a .bin file.
- Extracting coefficients from the given signal and comparing them with those in the trained models.
- Calculating and displaying the sound pressure level (SPL) of the given signal.
- Establishing a connection with the AGH University server via SSH.

## Project Contributor

The success of this project was greatly influenced by the contribution of the Bearded Agama, a remarkable animal.

![Alt text](bearded_agama.jpg?raw=true)

Thank you for your interest in "our" project! :dragon:
