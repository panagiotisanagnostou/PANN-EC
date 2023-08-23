from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection

import h5py
import os
import numpy as np
import os
import pickle
import tensorflow as tf


def h5file(fpath):
    name = os.path.basename(fpath).split(".")[0]
    print("\n" + name, end=" ")

    f = h5py.File(fpath, "r")
    inData = f["data"]["matrix"][:].transpose()
    inTarget = f["class"]["categories"][:]
    inTarget = np.int32(inTarget) - 1

    if inData.shape[0] != len(inTarget):
        inData = inData.transpose()
        if inData.shape[0] != len(inTarget):
            print("Data ", name, "error! Pls Check!")
            f.close()
            return
    f.close()

    return inData, inTarget


def boost_RP_train_set_PCA(  # dat,
    label,
    aug_rate=2,
    pca_dim=100,
    test_ratio=0.1,
    rp_error=0.1,
    test_idx=None,
    rseed=None,
):
    with open("tmp_file.dmp", "rb") as inf:
        dat = pickle.load(inf)

    np.random.seed(rseed)

    # aug_rate represent the multiplier of the data size (integer)
    d = johnson_lindenstrauss_min_dim(n_samples=dat.shape[0], eps=rp_error)
    transformer = GaussianRandomProjection(n_components=d * aug_rate)
    X_transformed = transformer.fit_transform(dat)
    n_samples = X_transformed.shape[0]
    dat = None

    # ============PCA stuff
    X_transformed = np.hsplit(X_transformed, aug_rate)
    X_PR = []
    pca = PCA(n_components=pca_dim, svd_solver="full")
    for i in range(0, aug_rate):
        X_PR.append(pca.fit_transform(X_transformed[i]))
    X_transformed = None
    # ============PCA stuff
    X_PCA_complete = np.concatenate(X_PR, axis=1)
    X_PR = None

    # I may need to use indexes in the future somehow
    indices = np.arange(n_samples)

    if test_idx is None:
        X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(
            X_PCA_complete, label, indices, test_size=test_ratio, random_state=rseed
        )
    else:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False

        X_train = X_PCA_complete[train_mask]
        X_test = X_PCA_complete[test_idx]

        y_train = label[train_mask]
        y_test = label[test_idx]

        idx2 = indices[test_idx]

    # here, for loop and append new data
    X_train = np.concatenate(np.hsplit(X_train, aug_rate), axis=0)
    X_test = np.concatenate(np.hsplit(X_test, aug_rate), axis=0)

    # edo kati ginete na to ftiakso
    y_train = np.tile(y_train, [aug_rate, 1])
    y_test = np.tile(y_test, [aug_rate, 1])

    y_test = np.concatenate(y_test, 0)
    y_train = np.concatenate(y_train, 0)

    # it will return new xtrain and xtest sets
    return X_train, X_test, y_train, y_test, d, idx2


def NN_training(X_train, y_train, aug_rate):
    tf.random.set_seed(123)

    # calculate the unique class labels (and their count)
    unq_values_train, counts_train = np.unique(y_train, return_counts=True)

    neural_network_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                units=100,
                activation="relu",
                input_shape=(X_train.shape[1],),
                name="FirstHiddenLayer",
            ),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=unq_values_train.shape[0], activation="softmax"
            ),
        ]
    )

    neural_network_model.compile(
        loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"]
    )

    # set batch size to count of train points per space
    batch_size = int(X_train.shape[0] / aug_rate)
    history = neural_network_model.fit(
        X_train,
        tf.keras.utils.to_categorical(y_train),
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        # validation_split=0.1,
    )

    return neural_network_model, history


def run_experiment(X, y, test_ratio=0.1, rp_error=0.1, rseed=None):
    iterations = 10
    pca_dim = 50

    results = {}
    all_labels = {}
    for aug_rate in [11]:  # [7, 11, 21]:
        labels = {}

        labels["nn_maj"] = np.empty((0, 2))
        labels["rf_maj"] = np.empty((0, 2))

        nn_acc_maj = []
        nn_acc_1st = []
        nn_acc_all = []
        nn_f1_maj = []
        nn_f1_1st = []
        nn_f1_all = []

        rf_acc_maj = []
        rf_acc_1st = []
        rf_acc_all = []
        rf_f1_maj = []
        rf_f1_1st = []
        rf_f1_all = []

        with open("tmp_file.dmp", "wb") as outf:
            pickle.dump(X, outf)

        print(f"Testing aug_rate={aug_rate}")
        for validations in range(iterations):
            (
                X_train,
                X_test,
                y_train,
                y_test,
                d,
                idx_test,
            ) = boost_RP_train_set_PCA(  # X,
                y,
                aug_rate=aug_rate,
                pca_dim=pca_dim,
                test_ratio=test_ratio,
                rp_error=rp_error,
                rseed=rseed,
            )

            print("Iteration", validations, "of 100")

            # Neural Network
            neural_network_model, history = NN_training(X_train, y_train, aug_rate)

            predictions_nn_probs = neural_network_model.predict(X_test)
            predictions_nn = np.argmax(predictions_nn_probs, axis=1)
            ens_res = np.column_stack(np.hsplit(predictions_nn, aug_rate))

            test_accuracy_nn = metrics.accuracy_score(y_test, predictions_nn)
            nn_acc_all.append(test_accuracy_nn)
            nn_f1_all.append(metrics.f1_score(y_test, predictions_nn, average="macro"))
            print(
                "The Accuracy of the Neural Network fall all data points (all versions)",
                test_accuracy_nn,
            )

            first_space = metrics.accuracy_score(
                y_test[0 : ens_res.shape[0]], ens_res[:, 0]
            )
            nn_acc_1st.append(first_space)
            nn_f1_1st.append(
                metrics.f1_score(
                    y_test[0 : ens_res.shape[0]], ens_res[:, 0], average="weighted"
                )
            )
            print(
                "The Accuracy of the Neural Network according to the first space only (1 replicate of the data)",
                first_space,
            )

            maj_result = stats.mode(ens_res, axis=1)
            if validations == 0:
                labels["nn_maj"] = np.concatenate(
                    (
                        labels["nn_maj"],
                        np.concatenate((idx_test[:, None], maj_result[0]), axis=1),
                    ),
                    axis=0,
                )
            else:
                labels["nn_maj"] = np.concatenate(
                    (
                        labels["nn_maj"],
                        np.concatenate((idx_test[:, None], maj_result[0]), axis=1),
                    ),
                    axis=1,
                )

            maj_accuracy_nn = metrics.accuracy_score(
                y_test[0 : ens_res.shape[0]], maj_result[0]
            )
            nn_acc_maj.append(maj_accuracy_nn)
            nn_f1_maj.append(
                metrics.f1_score(
                    y_test[0 : ens_res.shape[0]], maj_result[0], average="weighted"
                )
            )
            print(
                "The Accuracy of the Neural Network (embedding prediction)",
                maj_accuracy_nn,
            )

            # Random Forest
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            clf.fit(X_train, y_train)
            RandF_pred = clf.predict(X_test)

            test_accuracy_RF = metrics.accuracy_score(y_test, RandF_pred)
            rf_acc_all.append(test_accuracy_RF)
            rf_f1_all.append(metrics.f1_score(y_test, RandF_pred, average="weighted"))
            print(
                "The Accuracy of the RF for all data points (all versions)",
                test_accuracy_RF,
            )

            ens_res_RF = np.column_stack(np.hsplit(RandF_pred, aug_rate))
            maj_result_RF = stats.mode(ens_res_RF, axis=1)

            if validations == 0:
                labels["rf_maj"] = np.concatenate(
                    (
                        labels["rf_maj"],
                        np.concatenate((idx_test[:, None], maj_result_RF[0]), axis=1),
                    ),
                    axis=0,
                )
            else:
                labels["rf_maj"] = np.concatenate(
                    (
                        labels["rf_maj"],
                        np.concatenate((idx_test[:, None], maj_result_RF[0]), axis=1),
                    ),
                    axis=1,
                )

            maj_accuracy_RF = metrics.accuracy_score(
                y_test[0 : ens_res_RF.shape[0]], maj_result_RF[0]
            )
            rf_acc_maj.append(maj_accuracy_RF)
            rf_f1_maj.append(
                metrics.f1_score(
                    y_test[0 : ens_res_RF.shape[0]],
                    maj_result_RF[0],
                    average="weighted",
                )
            )
            print("The Accuracy of RF(embending prediction)", maj_accuracy_RF)

            # RF For a single data projection
            single_test_set = int(X_test.shape[0] / aug_rate)
            single_train_set = int(X_train.shape[0] / aug_rate)
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            clf.fit(X_train[0:single_train_set, :], y_train[0:single_train_set])
            RandF_pred_one = clf.predict(X_test[0:single_test_set, :])

            test_accuracy_RF_one = metrics.accuracy_score(
                y_test[0:single_test_set], RandF_pred_one
            )
            rf_acc_1st.append(test_accuracy_RF_one)
            rf_f1_1st.append(
                metrics.f1_score(
                    y_test[0:single_test_set], RandF_pred_one, average="weighted"
                )
            )
            print("The Accuracy of the RF one projection only", test_accuracy_RF_one)

            del X_train, X_test, y_train, y_test, d

        # Delete the tmp file
        if os.path.exists("tmp_file.dmp"):
            os.remove("tmp_file.dmp")
            print("The file 'tmp_file.dmp' has been deleted.")

        all_labels[aug_rate] = labels
        results[aug_rate] = {
            "nn_acc_all": np.mean(nn_acc_all),
            "nn_acc_all_std": np.std(nn_acc_all),
            "nn_f1_all": np.mean(nn_f1_all),
            "nn_f1_all_std": np.std(nn_f1_all),
            "nn_acc_1st": np.mean(nn_acc_1st),
            "nn_acc_1st_std": np.std(nn_acc_1st),
            "nn_f1_1st": np.mean(nn_f1_1st),
            "nn_f1_1st_std": np.std(nn_f1_1st),
            "nn_acc_maj": np.mean(nn_acc_maj),
            "nn_acc_maj_std": np.std(nn_acc_maj),
            "nn_f1_maj": np.mean(nn_f1_maj),
            "nn_f1_maj_std": np.std(nn_f1_maj),
            "rf_acc_all": np.mean(rf_acc_all),
            "rf_acc_all_std": np.std(rf_acc_all),
            "rf_f1_all": np.mean(rf_f1_all),
            "rf_f1_all_std": np.std(rf_f1_all),
            "rf_acc_1st": np.mean(rf_acc_1st),
            "rf_acc_1st_std": np.std(rf_acc_1st),
            "rf_f1_1st": np.mean(rf_f1_1st),
            "rf_f1_1st_std": np.std(rf_f1_1st),
            "rf_acc_maj": np.mean(rf_acc_maj),
            "rf_acc_maj_std": np.std(rf_acc_maj),
            "rf_f1_maj": np.mean(rf_f1_maj),
            "rf_f1_maj_std": np.std(rf_f1_maj),
        }

    return results, all_labels


def RNA_seq(filename):
    base_path = "data/"

    data_file = base_path + filename

    X, y = h5file(data_file)

    mat_dat = {"data": X, "class": y}

    del X, y

    idx = np.argwhere(np.all(mat_dat["data"][..., :] == 0, axis=0))
    dat_norm = np.delete(mat_dat["data"], idx, axis=1)

    print(mat_dat["data"].shape)
    dat_norm = normalize(dat_norm, axis=0, norm="max")
    labels = mat_dat["class"]
    results, labels = run_experiment(dat_norm, labels, test_ratio=0.1, rp_error=0.1)

    if not os.path.exists("scores/"):
        os.makedirs("scores/")

    with open("scores/" + filename + ".pkl", "wb") as f:
        pickle.dump(results, f)

    if not os.path.exists("labels/"):
        os.makedirs("labels/")

    with open("labels/" + filename + ".pkl", "wb") as f:
        pickle.dump(labels, f)

    return results
