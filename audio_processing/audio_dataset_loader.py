from typing import Dict, List, Tuple

import soundfile as sf
import os
import numpy as np
import librosa
import pickle
import shutil
from sklearn import mixture
import warnings
import matplotlib
import matplotlib.pyplot as plt


# function for drawing graphs of features extracted from specific audio recordings
def draw_graph() -> None:

    folder = "../pythonProject"
    file_path = "F:/engineer/output/2s_2048wl_512hl/bieganie/bieganie_08.wav"
    data, sr = sf.read(file_path)

    file_id = file_path.split("/")[-1].split(".")[-2]
    print(file_id)

    # signal graph
    time = np.arange(0, len(data) / sr, 1 / sr)
    plt.figure(1)
    plt.subplot(221)
    plt.plot(time, data)
    plt.grid()
    plt.xlim(left=0, right=len(data) / sr)
    plt.xlabel("Time [s]")
    plt.ylabel("")
    plt.savefig(f"{folder}/{file_id}.png")

    # fft
    freq = np.arange(0, int(len(data) / 2), 1) / 5
    S = np.abs(np.fft.fft(data))[0 : int(len(data) / 2)] / len(data)
    plt.subplot(222)
    plt.plot(freq, S)
    plt.grid()
    plt.xlim(left=0, right=sr / 2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")

    # zero crossing
    plt.figure(2)
    wek = np.arange(0, 2048) / 3.2594932345220164765467394738691
    wek = np.sin(wek)
    zero_sin = librosa.zero_crossings(wek, pad=False)
    print(sum(zero_sin))
    zero_sin_r = librosa.feature.zero_crossing_rate(wek)
    print(zero_sin_r.shape)
    plt.plot(zero_sin_r[0])
    plt.show()


flag = False
norm_vector = ""


def norm(row: np.ndarray) -> List[float]:
    # single row matrix
    normalised = list()
    global flag
    global norm_vector
    # if weights exist
    if flag:
        normalised = [i * j for i, j in zip(norm_vector, row.tolist()[0])]
    else:
        # if they dont
        flag = True
        before = row.tolist()[0]
        for x in row.tolist()[0]:
            while True:
                if x > 50 or x < -50:
                    x = x / 10
                elif x < 1 and x > -1:
                    x = x * 10
                else:
                    break
            normalised.append(x)
        norm_vector = [i / j for i, j in zip(normalised, before)]
    return normalised


# function for drawing histograms from specific audio recordings
def draw_hist(matrix_dict: List[np.ndarray], directory: str) -> None:

    names = ["bieganie", "jedzenie", "tlo"]
    row_test = matrix_dict[0][2]
    row_test = norm(row_test)
    print(np.shape(row_test))
    for i, matrix in enumerate(matrix_dict):
        plt.figure()
        row = np.mean(matrix, axis=0)
        print(np.shape(row))
        normalised = norm(row)
        distance = np.sqrt(
            sum([(i - j) * (i - j) for i, j in zip(normalised, row_test)])
        )
        print(distance)

        plt.bar(np.arange(0, 73), normalised, alpha=0.5, color="b")
        plt.bar(np.arange(0, 73), row_test, alpha=0.5, color="r")
        plt.grid()
        plt.xlim(left=-1, right=len(row_test))
        plt.xlabel("Numer cechy")
        plt.ylabel("Znormalizowana amplituda cechy")
        plt.legend(["Średnie wartości cech klasy", "Wartości cech próbki biegania"])

        plt.savefig(f"{directory}/{names[i]}.png")


class ModelRefiner:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        rewrite_data: bool,
        evaluation_percent: int,
    ):

        score_all_x_times_mean = dict()

        for durations in [1, 2, 3, 4, 5]:
            for win_length in [512, 1024, 2048]:
                for hop_length in [
                    int((1 / 4) * win_length),
                    int((1 / 2) * win_length),
                ]:
                    score_all_x_times = []
                    results_gmm_x_times = np.zeros((3, 3))
                    try:
                        warnings.filterwarnings("ignore")
                        ad = AudioDatasetLoader(
                            input_path=input_path,
                            output_path=output_path,
                            durations=durations,
                            window="typ",
                            win_length=win_length,
                            hop_length=hop_length,
                            rewrite_data=rewrite_data,
                            evaluation_percent=evaluation_percent,
                        )
                        matrix_activities = ad.load_matrixes()
                        draw_hist(matrix_activities, output_path)
                    except AssertionError:
                        continue

                    for c in range(10):
                        # DIVISION OF TRAINING AND TEST CHUNKS - two lists
                        matrixes_eval: List[np.ndarray] = []  # test/eval matrix
                        matrixes_train: List[np.ndarray] = []  # train matrix

                        for matrix in matrix_activities:
                            np.random.shuffle(matrix)  # matrix shuffle (20x73)
                            matrix_shape_x, y = np.shape(matrix)
                            matrix_shape_eval_x = int(
                                evaluation_percent / 100.0 * matrix_shape_x
                            )
                            matrixes_eval.append(matrix[0:matrix_shape_eval_x, :])
                            matrixes_train.append(matrix[matrix_shape_eval_x:, :])

                        trained_gmms = []  # list of 3 GMM models
                        for matrix in matrixes_train:
                            trained_gmms.append(self.train_gmm(matrix_eval=matrix, n=4, cov_type="diag"))
                        folder_path = (
                            output_path
                            + str(int(durations))
                            + "s_"
                            + str(win_length)
                            + "wl_"
                            + str(hop_length)
                            + "hl"
                        )
                        file = open(folder_path + "/" + "model.bin", "wb")
                        pickle.dump(trained_gmms, file)

                        # hint - when size is the 1st argument 2 parenthesis needed
                        results_gmm = np.zeros((3, 3))
                        correct_gmm = np.zeros((3, 1))

                        for matrix_list in [
                            matrixes_eval
                        ]:  # entering the list train/test
                            for count, matrix in enumerate(
                                matrix_list
                            ):  # entering the activity #enumerate
                                for row in matrix:  # entering the specific recording (features of data chunk)
                                    results_gmm_temp = []
                                    for gmm in trained_gmms:  # division for 3 activities (running, eating, acoustic background)
                                        results_gmm_temp.append(int(gmm.score(row)))
                                    if np.argmax(results_gmm_temp) == count:  # counter of correct classification
                                        correct_gmm[count] += 1
                                    results_gmm[count, np.argmax(results_gmm_temp)] += 1  # counter of all classifications
                            results_gmm_x_times = results_gmm_x_times + results_gmm

                        with open(folder_path + "/" + "Wyniki.txt", "w+") as file:
                            file.write("Macierz:\n")
                            score_all.append(
                                "Macierz:"
                                + str(int(durations))
                                + "s_"
                                + str(win_length)
                                + "wl_"
                                + str(hop_length)
                                + "hl"
                                + "\n"
                            )
                            for row in results_gmm:
                                file.write(str(row) + "\n")
                                score_all.append(str(row))

                            score_all_x_times.append(
                                np.sum(correct_gmm.T / np.sum(results_gmm, axis=1)) / 3
                            )
                            file.write("Sprawność:")
                            file.write(str(correct_gmm / np.sum(results_gmm)) + "\n")
                            file.close()

                    results_gmm_x_times = results_gmm_x_times / how_many_times
                    score_all_x_times_mean[
                        str(int(durations))
                        + "s_"
                        + str(win_length)
                        + "wl_"
                        + str(hop_length)
                        + "hl"
                    ] = [
                        str(sum(score_all_x_times) / how_many_times),
                        str(np.round(results_gmm_x_times, 1)),
                    ]

        with open(output_path + "wyniki_wszystkie_x_times.txt", "w+") as file:
            for key, value in score_all_x_times_mean.items():
                file.write(str(key))
                file.write("\n")
                file.write("\n".join(value))
                file.write("\n\n")

        w = [[key, value] for key, value in score_all_x_times_mean.items()]
        results_sorted = sorted(w, key=lambda l: l[1][0], reverse=True)
        print(results_sorted[0:3])

    def train_gmm(self, matrix_eval: np.ndarray, n: int, cov_type: str) -> mixture.GaussianMixture:
        gmm = mixture.GaussianMixture(
            n_components=n,
            covariance_type=cov_type,
            covar=1e-06,
            tol=0.001,
            reg_max_iter=100,
            n_init=1,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose=3,
            verbose_interval=10,
        )
        gmm.fit(matrix_eval)
        return gmm


class AudioDatasetLoader:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        durations: int,
        window: str,
        hop_length: int,
        win_length: int,
        rewrite_data: bool,
        evaluation_percent: int,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.durations = durations
        self.activities = ["bieganie", "jedzenie", "tlo"]
        self.window = window
        self.hop_length = hop_length
        self.win_length = win_length
        self.rewrite_data = rewrite_data
        self.evaluation_percent = evaluation_percent

    def extract_coefficient(self, data_chunk: np.ndarray, sr: int) -> np.array:

        features = []
        y = data_chunk
        Y = abs(librosa.stft(y, hop_length=self.hop_length, win_length=self.win_length))

        features.append(librosa.feature.zero_crossing_rate(y))  # zero_crossing
        features.append(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )  # spectral_centroids
        features.append(
            librosa.feature.spectral_bandwidth(y=y, sr=sr, p=2)
        )  # spectral_bandwidth 1
        features.append(
            librosa.feature.spectral_bandwidth(y=y, sr=sr, p=3)
        )  # spectral_bandwidth 2
        features.append(
            librosa.feature.spectral_bandwidth(y=y, sr=sr, p=4)
        )  # spectral_bandwidth 3
        features.append(
            librosa.feature.spectral_rolloff(y=y, sr=sr)
        )  # spectral_rolloff
        features.append(librosa.feature.spectral_flatness(y))  # flatness
        features.append(librosa.feature.rms(y))  # rms
        features.append(
            librosa.feature.poly_features(featuresS=Y, order=0)[0]
        )  # poly_features_0
        features.append(
            librosa.feature.poly_features(featuresS=Y, order=1)[1]
        )  # poly_features_1
        features.append(
            librosa.feature.poly_features(featuresS=Y, order=2)[2]
        )  # poly_features_2
        features.append(
            librosa.feature.poly_features(featuresS=Y, order=3)[3]
        )  # poly_features_3
        features.append(
            librosa.feature.poly_features(featuresS=Y, order=4)[4]
        )  # poly_features_4

        for i in range(len(features)):
            features[i] = np.mean(features[i])

        # mfcc feature extraction
        temp = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20,
            lifter=0,
            hop_length=self.hop_length,
            win_length=self.win_length,
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
        features = np.array(features)

        return features

    def process(self) -> None:

        folder_path = (
            self.output_path
            + str(int(self.durations))
            + "s_"
            + str(self.win_length)
            + "wl_"
            + str(self.hop_length)
            + "hl"
        )
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        else:
            try:
                shutil.rmtree(folder_path)
                os.mkdir(folder_path)
                os.mkdir(folder_path)
            except:
                print("Error while deleting directory")

        features_dict = dict()  # initialization of empty dictionary

        for act in self.activities:

            os.mkdir(folder_path + "/" + act)
            data, sr = sf.read(self.input_path + act + ".wav")
            number_of_chunks = np.floor(len(data) / sr / self.durations)
            features_dict[act] = dict()

            for i in range(int(number_of_chunks)):
                data_chunk = data[np.floor(self.durations * i * sr): np.floor(self.durations * (i + 1) * sr)]
                sf.write(
                    folder_path
                    + "/"
                    + act
                    + "/"
                    + act
                    + "_"
                    + str("{:0>2}".format(i))
                    + ".wav",
                    data_chunk,
                    sr,
                )

                features = self.extract_coefficient(data_chunk=data_chunk, sr=sr)
                features_dict[act][str(i).rjust(2, "0")] = features

        file = open(folder_path + "/" + "cechy.bin", "wb")
        pickle.dump(features_dict, file)
        file.close()

    def load_matrixes(self) -> List[np.ndarray]:

        folder_path = (
            self.output_path
            + str(int(self.durations))
            + "s_"
            + str(self.win_length)
            + "wl_"
            + str(self.hop_length)
            + "hl"
        )

        if not os.path.isdir(folder_path) or self.rewrite_data:
            self.process()

        features_dict_loaded = pickle.load(open(folder_path + "/cechy.bin", "rb"))
        matrix_activities = []

        # items - returning list of [key, value]
        for act in features_dict_loaded.keys():  # repeating 3 times
            # reading number of features from "deeper" dictionary, 0 - first pair, 1 - value of first pair
            number_of_features = len(list(features_dict_loaded[act].values())[0])
            # hint - matrix 73x0 - empty list "anchor" ("kotwica") - able to add something to it, but not to write inside
            matrix_eval = np.ndarray(shape=[0, number_of_features])

            assert (
                1.0 / len(features_dict_loaded[act]) <= self.evaluation_percent / 100.0
            ), "Za mało nagrań, minimalny procent wynosi " + str(
                100 / len(features_dict_loaded[act])
            )

            for key, value in features_dict_loaded[act].items():
                if np.shape(matrix_eval)[0] == 600 / self.durations:
                    continue
                matrix_eval = np.concatenate((matrix_eval, np.matrix(value)), axis=0)

            matrix_activities.append(matrix_eval)

        return matrix_activities
