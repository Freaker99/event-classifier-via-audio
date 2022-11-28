from typing import Dict, List

import time
import numpy as np
import soundfile as sf
import librosa
import pickle
from sklearn import mixture
import os
import matplotlib.pyplot as plt
import paramiko


def load_model(folder_path_system: str) -> List[mixture.GaussianMixture]:
    model = pickle.load(open(folder_path_system + "model.bin", "rb"))
    # model - list of three GMM models
    return model


def load_wav(folder_path_system: str, file_name: str) -> np.matrix:
    folder_path_audio = folder_path_system + "audio/"
    data_chunk, sr = sf.read(folder_path_audio + file_name)
    features = extract_coefficient(data_chunk=data_chunk, sr=sr)
    return features


def extract_coefficient(data_chunk: np.ndarray, sr: int) -> np.matrix:
    features = []
    hop_length = 128  # hop length
    win_length = 512  # window length
    y = data_chunk
    Y = abs(librosa.stft(y, hop_length=hop_length, win_length=win_length))

    features.append(librosa.feature.zero_crossing_rate(y))  # zero_crossing
    features.append(librosa.feature.spectral_centroid(y=y, sr=sr))  # spectral_centroids
    features.append(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, p=2)
    )  # spectral_bandwidth 1
    features.append(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, p=3)
    )  # spectral_bandwidth 2
    features.append(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, p=4)
    )  # spectral_bandwidth 3
    features.append(librosa.feature.spectral_rolloff(y=y, sr=sr))  # spectral_rolloff
    features.append(librosa.feature.spectral_flatness(y))  # flatness
    features.append(librosa.feature.rms(y))  # rms
    features.append(librosa.feature.poly_features(S=Y, order=0)[0])  # poly_features_0
    features.append(librosa.feature.poly_features(S=Y, order=1)[1])  # poly_features_1
    features.append(librosa.feature.poly_features(S=Y, order=2)[2])  # poly_features_2
    features.append(librosa.feature.poly_features(S=Y, order=3)[3])  # poly_features_3
    features.append(librosa.feature.poly_features(S=Y, order=4)[4])  # poly_features_4

    for i in range(len(features)):
        features[i] = np.mean(features[i])
    # mfcc feature extraction
    temp = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=20, lifter=0, hop_length=hop_length, win_length=win_length
    )

    # list of features, addition
    features = features + list(np.mean(temp, axis=1))

    features = features + list(
        np.mean(librosa.feature.delta(temp, order=1), axis=1)
    )  # mfcc first derivative
    features = features + list(
        np.mean(librosa.feature.delta(temp, order=2), axis=1)
    )  # mfcc second derivative

    # conversion to matrix (vector of features Nx1)
    features = np.matrix(features)

    return features


def draw_graph(folder_path_system: str, file_name: str) -> None:
    folder_path_audio = folder_path_system + "audio/"
    folder_path_graph = folder_path_system + "wykresy/"
    # file read
    data, sr = sf.read(folder_path_audio + file_name)

    # hint - (-1) - last element of the list, (-2) - penultimate ... etc.
    file_id = file_name.split(".")[-2]
    print(file_id)

    # signal graph
    time = np.arange(0, len(data) / sr, 1 / sr)
    plt.figure(1)
    plt.clf()
    plt.plot(time, data)
    plt.grid()
    plt.xlim(left=0, right=len(data) / sr)
    plt.xlabel("Time [s]")
    plt.ylabel("")
    plt.savefig(f"{folder_path_graph}/{file_id}.png")
    plt.clf()


def spl(folder_path_system: str, file_name: str) -> float:
    folder_path_audio = folder_path_system + "audio/"
    # file read
    data, sr = sf.read(folder_path_audio + file_name)

    mean_pressure = np.mean(np.multiply(data, data))
    spl = 20 * (np.log10(mean_pressure * 1000000000))
    return spl


def draw_spl(
    folder_path_system: str,
    file_name: str,
    spl_list: List[float],
    recent_file_name_list: List[str],
) -> None:
    folder_path_graph = folder_path_system + "wykresy/"
    file_id = file_name.split(".")[-2]
    print(file_id)

    file_names = recent_file_name_list
    plt.figure(1)
    plt.clf()
    plt.bar(file_names, spl_list)
    plt.grid()
    plt.xticks(rotation=30)
    plt.xlabel("Data i godzina")
    plt.ylabel("SPL [dB]")
    plt.savefig(f"{folder_path_graph}/{file_id}_SPL.png", bbox_inches="tight")


def connect_ssh() -> paramiko.SSHClient:
    server = "***"
    username = "***"
    password = "***"
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password, look_for_keys=False)
    return ssh


def sent_file_ssh(ssh: paramiko.SSHClient) -> None:
    localpath = "F:/engineer/test/system/wykresy/bieganie_00.png"
    remotepath = "public_html/wykresy"
    sftp = ssh.open_sftp()
    sftp.put(localpath, remotepath)
    sftp.close()
    ssh.close()


def main() -> None:
    folder_path_system = "F:/engineer/test/system/"
    # ssh = connect_ssh()
    spl_list = []
    recent_file_name_list = []
    trained_gmms = load_model(folder_path_system)
    activities = ["bieganie", "jedzenie", "tlo"]
    audio_path = ""
    while True:
        audio_list = os.listdir(folder_path_system + "audio/")
        audio_list = sorted(audio_list)
        if len(audio_list) == 0:
            time.sleep(0.1)
            continue
        if audio_path == audio_list[0]:
            time.sleep(0.1)
            continue
        audio_path = audio_list[0]  # name of the file .wav

        # features of the imported file
        features = load_wav(folder_path_system, audio_path)
        results_gmm_temp = []
        for gmm in trained_gmms:
            results_gmm_temp.append(int(gmm.score(features)))
        act = activities[np.argmax(results_gmm_temp)]

        with open(folder_path_system + "logs.txt", "a+") as file:
            file.write("plik:\t" + audio_path + "\t" + "aktywność:\t" + act + "\n")
            file.close()

        spl_list.append(spl(folder_path_system, audio_path))
        recent_file_name_list.append(audio_path.rsplit(".", maxsplit=1)[0])
        os.rename(
            folder_path_system + "audio/" + audio_path,
            folder_path_system + "audio_rozpoznane/" + audio_path,
        )
        if len(spl_list) < 5:
            continue
        elif len(spl_list) > 12:
            spl_list = spl_list[:12]
            recent_file_name_list = recent_file_name_list[:12]
        draw_spl(folder_path_system, audio_path, spl_list, recent_file_name_list)
        # sent_file_ssh(ssh)


if __name__ == "__main__":
    main()
