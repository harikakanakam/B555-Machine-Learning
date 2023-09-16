import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

all_train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def cal_gen_error_rate(train_full, labels_train_full, dt_train_size, test, labels_test):
    runs = 30
    mean_error_rate = []
    sd_error_rate = []
    for each_train_size in all_train_sizes:
        sum_error_rate = 0
        error_rates = []
        for each_run in range(runs):
            train_size = round(each_train_size * dt_train_size)
            train = train_full.head(train_size)
            train_labels = labels_train_full.head(train_size)
            m1_sum_tx = 0
            m2_sum_tx = 0
            N1 = train_labels[0].sum()
            N2 = train_size - N1
            N = N1 + N2
            train = train.reset_index()
            for index, each_eg in train.iterrows():
                each_eg = each_eg[1:]
                tx = train_labels.iloc[index][0]
                if tx == 1:
                    m1_sum_tx += each_eg * tx
                else:
                    m2_sum_tx += each_eg * (1 - tx)

            mu1 = m1_sum_tx / N1
            mu2 = m2_sum_tx / N2

            sum_s1 = 0
            sum_s2 = 0
            for index, each_eg in train.iterrows():
                each_eg = each_eg[1]
                tx = train_labels.iloc[index][0]
                if tx == 1:
                    x_mu1 = each_eg - mu1
                    sum_s1 += np.dot(x_mu1, np.transpose(x_mu1))
                else:
                    x_mu2 = each_eg - mu2
                    sum_s2 += np.dot(x_mu2, np.transpose(x_mu2))

            S = (sum_s1 + sum_s2) / N
            mu_diff = (mu1 - mu2)

            w = S * mu_diff
            w_df = pd.DataFrame(w)

            w0 = (-0.5 * np.dot(np.transpose(mu1), np.dot(S, mu1))) + (
                    0.5 * np.dot(np.transpose(mu2), np.dot(S, mu2))) + math.log(N1 / N2)
            wt_x = (np.dot(test, w_df) + w0)

            P = []
            cal_label = []
            for each in wt_x:
                try:
                    val = 1 / (1 + math.pow(math.e, -each))
                except OverflowError:
                    val = 0
                P.append(val)
                cal_label.append(1 if val >= 0.5 else 0)
            error_sum = 0
            for i in range(len(cal_label)):
                error_sum += 1 if cal_label[i] != labels_test.iloc[i][0] else 0
            error_rate = error_sum / len(cal_label)
            sum_error_rate += error_rate
            error_rates.append(error_rate)
        mean_error_rate.append(np.mean(error_rates))
        sd_error_rate.append(np.std(error_rates))
    result = pd.DataFrame({"Train Size": all_train_sizes, "Mean Error Rate": mean_error_rate,
                           "SD error rate": sd_error_rate})
    print(result)
    return mean_error_rate, sd_error_rate


def generative_model():
    A_data = pd.read_csv('pp3data/A.csv', header=None)
    A_labels_data = pd.read_csv('pp3data/labels-A.csv', header=None)
    A_size = A_data.shape[0]
    A_train_size = round(2 * A_size / 3)
    A_test_size = round(1 * A_size / 3)
    A_train_full = A_data.tail(A_train_size)
    A_labels_train_full = A_labels_data.tail(A_train_size)
    A_test = A_data.head(A_test_size)
    A_labels_test = A_labels_data.head(A_test_size)
    print("*********************Error rates for the A Dataset***********************")
    A_mean, A_std = cal_gen_error_rate(A_train_full, A_labels_train_full, A_train_size, A_test, A_labels_test)

    B_data = pd.read_csv('pp3data/B.csv', header=None)
    B_labels_data = pd.read_csv('pp3data/labels-B.csv', header=None)
    B_size = B_data.shape[0]
    B_train_size = round(2 * B_size / 3)
    B_test_size = round(1 * B_size / 3)
    B_train_full = B_data.tail(B_train_size)
    B_labels_train_full = B_labels_data.tail(B_train_size)
    B_test = B_data.head(B_test_size)
    B_labels_test = B_labels_data.head(B_test_size)
    print("*********************Error rates for the B Dataset***********************")
    B_mean, B_std = cal_gen_error_rate(B_train_full, B_labels_train_full, B_train_size, B_test, B_labels_test)

    usps_data = pd.read_csv('pp3data/usps.csv', header=None)
    usps_labels_data = pd.read_csv('pp3data/labels-usps.csv', header=None)
    usps_size = usps_data.shape[0]
    usps_train_size = round(2 * usps_size / 3)
    usps_test_size = round(1 * usps_size / 3)
    usps_train_full = usps_data.tail(usps_train_size)
    usps_labels_train_full = usps_labels_data.tail(usps_train_size)
    usps_test = usps_data.head(usps_test_size)
    usps_labels_test = usps_labels_data.head(usps_test_size)
    print("*********************Error rates for the USPS Dataset***********************")
    usps_mean, usps_sd = cal_gen_error_rate(
        usps_train_full, usps_labels_train_full, usps_train_size, usps_test, usps_labels_test)

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    x = np.array(all_train_sizes)
    ya_mean = np.array(A_mean)
    ya_sd = np.array(A_std)
    yb_mean = np.array(B_mean)
    yb_sd = np.array(B_std)
    yusps_mean = np.array(usps_mean)
    yusps_std = np.array(usps_sd)
    axis[0].plot(x, ya_mean)
    axis[0].plot(x, ya_sd)
    axis[0].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of A dataset")
    axis[0].legend(["A Mean", "A Std"])

    axis[1].plot(x, yb_mean)
    axis[1].plot(x, yb_sd)
    axis[1].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of B dataset")
    axis[1].legend(["B Mean", "B Std"])

    axis[2].plot(x, yusps_mean)
    axis[2].plot(x, yusps_std)
    axis[2].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of usps dataset")
    axis[2].legend(["usps Mean", "usps Std"])

    plt.savefig(f"Task 1_a.png")


def bayesian_model():
    A_data = pd.read_csv('pp3data/A.csv', header=None)
    A_labels_data = pd.read_csv('pp3data/labels-A.csv', header=None)
    A_size = A_data.shape[0]
    A_fea_count = A_data.shape[1]
    one_data = [1 for i in range(A_size)]
    A_data.insert(loc=0, column=A_fea_count + 1, value=one_data)
    A_train_size = round(2 * A_size / 3)
    A_test_size = round(1 * A_size / 3)
    A_train_full = A_data.tail(A_train_size)
    A_labels_train_full = A_labels_data.tail(A_train_size)
    A_test = A_data.head(A_test_size)
    A_labels_test = A_labels_data.head(A_test_size)
    print("*********************Error rates for the A Dataset***********************")
    A_mean, A_sd = call_bayesian_errors(A_fea_count, A_train_size, A_train_full, A_labels_train_full, A_test,
                                        A_labels_test)

    B_data = pd.read_csv('pp3data/B.csv', header=None)
    B_labels_data = pd.read_csv('pp3data/labels-B.csv', header=None)
    B_size = B_data.shape[0]
    B_fea_count = B_data.shape[1]
    one_data = [1 for i in range(B_size)]
    B_data.insert(loc=0, column=B_fea_count + 1, value=one_data)
    B_train_size = round(2 * B_size / 3)
    B_test_size = round(1 * B_size / 3)
    B_train_full = B_data.tail(B_train_size)
    B_labels_train_full = B_labels_data.tail(B_train_size)
    B_test = B_data.head(B_test_size)
    B_labels_test = B_labels_data.head(B_test_size)
    print("*********************Error rates for the B Dataset***********************")
    B_mean, B_sd = call_bayesian_errors(B_fea_count, B_train_size, B_train_full, B_labels_train_full, B_test,
                                        B_labels_test)

    usps_data = pd.read_csv('pp3data/usps.csv', header=None)
    usps_labels_data = pd.read_csv('pp3data/labels-usps.csv', header=None)
    usps_size = usps_data.shape[0]
    usps_fea_count = usps_data.shape[1]
    one_data = [1 for i in range(usps_size)]
    usps_data.insert(loc=0, column=usps_fea_count + 1, value=one_data)
    usps_train_size = round(2 * usps_size / 3)
    usps_test_size = round(1 * usps_size / 3)
    usps_train_full = usps_data.tail(usps_train_size)
    usps_labels_train_full = usps_labels_data.tail(usps_train_size)
    usps_test = usps_data.head(usps_test_size)
    usps_labels_test = usps_labels_data.head(usps_test_size)
    print("*********************Error rates for the USPS Dataset***********************")
    usps_mean, usps_sd = call_bayesian_errors(usps_fea_count, usps_train_size, usps_train_full, usps_labels_train_full,
                                              usps_test, usps_labels_test)

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    x = np.array(all_train_sizes)
    ya_mean = np.array(A_mean)
    ya_sd = np.array(A_sd)
    yb_mean = np.array(B_mean)
    yb_sd = np.array(B_sd)
    yusps_mean = np.array(usps_mean)
    yusps_std = np.array(usps_sd)
    axis[0].plot(x, ya_mean)
    axis[0].plot(x, ya_sd)
    axis[0].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of A dataset")
    axis[0].legend(["A Mean", "A Std"])

    axis[1].plot(x, yb_mean)
    axis[1].plot(x, yb_sd)
    axis[1].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of B dataset")
    axis[1].legend(["B Mean", "B Std"])

    axis[2].plot(x, yusps_mean)
    axis[2].plot(x, yusps_std)
    axis[2].set(xlabel="Size", ylabel="Test error",
                title=f"Test Error rate of usps dataset")
    axis[2].legend(["usps Mean", "usps Std"])
    plt.savefig(f"Task 1_b.png")


def call_bayesian_errors(fea_count, dt_train_size, train_full, labels_train_full, test, labels_test):
    runs = 30
    mean_error_rate = []
    sd_error_rate = []
    for each_train_size in all_train_sizes:
        error_rates = []
        for i in range(runs):
            train_size = round(each_train_size * dt_train_size)
            train = train_full.head(train_size)
            train_labels = labels_train_full.head(train_size)
            w, sn, w_list, time_diff = cal_bayesian_weights(fea_count + 1, train, train_labels)
            error_rate = cal_bayesian_error_rate(w, sn, test, labels_test)
            error_rates.append(error_rate)
        mean_error_rate.append(np.mean(error_rates))
        sd_error_rate.append(np.std(error_rates))
    result = pd.DataFrame({"Train Size": all_train_sizes, "Mean Error Rate": mean_error_rate,
                           "Std Error Rates": sd_error_rate})
    print(result)
    return mean_error_rate, sd_error_rate


def cal_bayesian_weights(fea_count, train, train_labels):
    alpha = 0.1
    n = 1
    w = pd.DataFrame([[0 for i in range(fea_count)]])
    list_w = []
    list_time_diff = []
    current_time = datetime.now()
    while n <= 100:
        list_w.append(w)
        a = np.dot(w, np.transpose(train))
        y = 1 / (1 + np.exp(-a))
        R = np.diag(np.array((y * (1 - y))[0]))
        phi_trans_R_phi = np.dot(np.transpose(train), np.dot(R, train))
        alpha_phi_trans_R_phi = alpha + phi_trans_R_phi
        stop_cond = None
        if np.linalg.det(alpha_phi_trans_R_phi) != 0:
            t1 = np.linalg.inv(alpha_phi_trans_R_phi)

            alpha_w = alpha * np.transpose(w)
            phi_trans_yt = np.dot(np.transpose(train), (np.transpose(y) - train_labels))
            t2 = phi_trans_yt + alpha_w

            t3 = np.dot(t1, t2)
            w_new = w - np.transpose(t3)
            num_norm = np.linalg.norm(w_new - w)
            den_norm = np.linalg.norm(w)
            if den_norm != 0:
                stop_cond = num_norm / den_norm
            w = w_new
        list_time_diff.append((datetime.now() - current_time).total_seconds())
        if stop_cond is not None and stop_cond < 0.0001:
            break
        n += 1
    sn_val = np.dot(np.transpose(train), np.dot(R, train))
    alpha_identity = alpha * np.identity(fea_count)
    sn = np.linalg.inv(sn_val + alpha_identity)
    return w, sn, list_w, list_time_diff


def cal_bayesian_error_rate(w, sn, test, labels_test):
    p_list = []
    cal_label = []
    for index, each_eg in test.iterrows():
        mu = np.dot(w, np.transpose(each_eg))
        var = np.dot(np.dot(each_eg, sn), np.transpose(each_eg))
        k_denom = 1 + (3.14 * var / 8)
        k = 1 / np.sqrt(k_denom)
        kmu = np.dot(k, mu)
        try:
            p = 1 / (1 + math.exp(-kmu))
        except OverflowError:
            p = math.inf

        p_list.append(p)
        cal_label.append(1 if p >= 0.5 else 0)
    error_sum = 0
    for i in range(len(cal_label)):
        error_sum += 1 if cal_label[i] != labels_test.iloc[i][0] else 0
    error_rate = error_sum / len(cal_label)
    return error_rate


def cal_task2_blr_error_mean(fea_count, train_size, train_full, labels_train_full, test_size, test, labels_test):
    runs = 3
    error_rates = []
    one_data = [1 for i in range(train_size)]
    train_full.insert(loc=0, column=fea_count + 1, value=one_data)
    train = train_full
    train_labels = labels_train_full
    one_data = [1 for i in range(test_size)]
    test.insert(loc=0, column=fea_count + 1, value=one_data)
    sn = None
    all_time_diff = []
    w_list = []
    for i in range(runs):
        w, sn, w_list, time_diff = cal_bayesian_weights(fea_count + 1, train, train_labels)
        all_time_diff.append(time_diff)
    avg_time_diff = np.mean(all_time_diff, axis=0)
    for each_w in w_list:
        error_rate = cal_bayesian_error_rate(each_w, sn, test, labels_test)
        error_rates.append(error_rate)
    result = pd.DataFrame({"Avg Time Diff": avg_time_diff, "Error Rates": error_rates})
    print(result)
    return avg_time_diff, error_rates


def cal_ga(fea_count, train_size, train, labels_train, test, labels_test):
    runs = 3
    all_time_diff = []
    list_w = []
    sn = None
    for i in range(runs):
        sn, list_w, time_diff = calculate_gradient_weights(fea_count, train_size, train, labels_train)
        all_time_diff.append(time_diff)
    avg_time_diff = np.mean(all_time_diff, axis=0)
    error_rates = []
    for each_w in list_w:
        error_rate = cal_bayesian_error_rate(each_w, sn, test, labels_test)
        error_rates.append(error_rate)
    result = pd.DataFrame({"Avg Time Diff": avg_time_diff, "Error Rates": error_rates})
    print(result)
    return avg_time_diff, error_rates


def calculate_gradient_weights(fea_count, dt_train_size, train_full, labels_train_full):
    alpha = 0.1
    train_size = dt_train_size
    train = train_full.head(train_size)
    train_labels = labels_train_full.head(train_size)
    n = 1
    w = pd.DataFrame([[0 for i in range(fea_count)]])
    eta = 0.0001
    start_time = datetime.now()
    c = 0
    list_w = []
    list_time_diff = []
    while n <= 6000:
        c += 1
        a = np.dot(w, np.transpose(train))
        y = 1 / (1 + np.exp(-a))
        R = np.diag(np.array((y * (1 - y))[0]))
        alpha_w = alpha * np.transpose(w)
        phi_trans_yt = np.dot(np.transpose(train), (np.transpose(y) - train_labels))
        t1 = phi_trans_yt + alpha_w
        t2 = eta * t1
        w_new = w - np.transpose(t2)
        num_norm = np.linalg.norm(w_new - w)
        den_norm = np.linalg.norm(w)
        stop_cond = None
        if den_norm != 0:
            stop_cond = num_norm / den_norm
        w = w_new
        n += 1
        if c == 10:
            time_diff = (datetime.now() - start_time).total_seconds()
            list_w.append(w)
            list_time_diff.append(time_diff)
            c = 1
        if stop_cond is not None and stop_cond < 0.0001:
            break
    sn_val = np.dot(np.transpose(train), np.dot(R, train))
    alpha_identity = alpha * np.identity(fea_count)
    sn = np.linalg.inv(sn_val + alpha_identity)
    return sn, list_w, list_time_diff


def cal_task2():
    A_data = pd.read_csv('pp3data/A.csv', header=None)
    A_labels_data = pd.read_csv('pp3data/labels-A.csv', header=None)
    A_size = A_data.shape[0]
    A_fea_count = A_data.shape[1]

    A_train_size = round(2 * A_size / 3)
    A_test_size = round(1 * A_size / 3)
    A_train_full = A_data.head(A_train_size)
    A_labels_train_full = A_labels_data.head(A_train_size)
    A_test = A_data.tail(A_test_size)
    A_labels_test = A_labels_data.tail(A_test_size)
    print("**************** Gradient Ascent Summary for dataset A ******************")
    A_ga_time_diff, A_ga_error_rate = cal_ga(A_fea_count, A_train_size, A_train_full, A_labels_train_full, A_test,
                                             A_labels_test)
    print("**************** BLR Summary for dataset A ******************")
    A_blr_time_diff, A_blr_error_rate = cal_task2_blr_error_mean(A_fea_count, A_train_size, A_train_full,
                                                                 A_labels_train_full, A_test_size, A_test,
                                                                 A_labels_test)

    usps_data = pd.read_csv('pp3data/usps.csv', header=None)
    usps_labels_data = pd.read_csv('pp3data/labels-usps.csv', header=None)
    usps_size = usps_data.shape[0]
    usps_fea_count = usps_data.shape[1]

    usps_train_size = round(2 * usps_size / 3)
    usps_test_size = round(1 * usps_size / 3)
    usps_train_full = usps_data.head(usps_train_size)
    usps_labels_train_full = usps_labels_data.head(usps_train_size)
    usps_test = usps_data.tail(usps_test_size)
    usps_labels_test = usps_labels_data.tail(usps_test_size)

    print("**************** Gradient Ascent Summary for dataset USPS ******************")
    usps_ga_time_diff, usps_ga_error_rate = cal_ga(usps_fea_count, usps_train_size, usps_train_full,
                                                   usps_labels_train_full, usps_test, usps_labels_test)

    print("**************** BLR Summary for dataset USPS ******************")
    usps_blr_time_diff, usps_blr_error_rate = cal_task2_blr_error_mean(
        usps_fea_count, usps_train_size, usps_train_full, usps_labels_train_full, usps_test_size,
        usps_test, usps_labels_test)

    fig, axis = plt.subplots(2, 2, figsize=(15, 15))
    xa_blr = np.array(A_blr_time_diff)
    ya_err_blr = np.array(A_blr_error_rate)
    xa_ga = np.array(A_ga_time_diff)
    ya_err_ga = np.array(A_ga_error_rate)

    axis[0][0].plot(xa_blr, ya_err_blr)
    axis[0][0].set(xlabel="Time difference", ylabel="Test error",
                   title=f"A dataset BLR")
    axis[0][0].legend(["Test Error"])

    axis[0][1].plot(xa_ga, ya_err_ga)
    axis[0][1].set(xlabel="Time difference", ylabel="Test error",
                   title=f"A dataset GA")
    axis[0][1].legend(["Test Error"])

    xu_blr = np.array(usps_blr_time_diff)
    yu_err_blr = np.array(usps_blr_error_rate)
    xu_ga = np.array(usps_ga_time_diff)
    yu_err_ga = np.array(usps_ga_error_rate)

    axis[1][0].plot(xu_blr, yu_err_blr)
    axis[1][0].set(xlabel="Time difference", ylabel="Test error",
                   title=f"USPS dataset BLR")
    axis[1][0].legend(["Test Error"])

    axis[1][1].plot(xu_ga, yu_err_ga)
    axis[1][1].set(xlabel="Time difference", ylabel="Test error",
                   title=f"USPS dataset GA")
    axis[1][1].legend(["Test Error"])

    plt.savefig(f"Task 2.png")


def task1():
    generative_model()
    bayesian_model()


def task2():
    cal_task2()


task1()
task2()
