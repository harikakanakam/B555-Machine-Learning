import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy
import random


f = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]


def calculate_weight(lambda_val, train, train_target):
    train_phi_transpose = numpy.transpose(train)
    phi_t_phi = numpy.dot(train_phi_transpose, train)

    identity = numpy.identity(phi_t_phi.shape[0])
    lambda_identity = lambda_val * identity

    w1 = lambda_identity + phi_t_phi
    inv_w1 = numpy.linalg.inv(w1)

    w2 = numpy.dot(train_phi_transpose, train_target)

    w = numpy.dot(inv_w1, w2)
    return w


def calculate_mse(w, test, test_target):
    y = numpy.dot(test, w)
    y_t_diff = y - test_target
    y_t_diff_square = numpy.square(y_t_diff)
    N = y_t_diff_square.shape[0]
    sum = y_t_diff_square.sum()
    mse = sum[0]/N
    # print(f"mse: {mse}")
    return mse


def mle(train, train_target, test, test_target, lambda_val):
    w = calculate_weight(lambda_val, train, train_target)
    mse = calculate_mse(w, test, test_target)
    return mse


def model_selection(train, train_target):
    old_alpha = random.randrange(1, 10)
    old_beta = random.randrange(1, 10)
    phi_transpose = numpy.transpose(train)
    phi_t_phi = numpy.dot(phi_transpose, train)
    identity = numpy.identity(phi_t_phi.shape[0])

    while True:
        beta_phi_t_phi = old_beta * phi_t_phi
        alpha_identity = old_alpha * identity
        inv_sn = alpha_identity + beta_phi_t_phi
        sn = numpy.linalg.inv(inv_sn)
        # print(f"sn: {sn.shape}")

        m1 = old_beta * sn
        m2 = numpy.dot(phi_transpose, train_target)
        mn = numpy.dot(m1, m2)
        # print(f"mn: {mn.shape}")

        eigen = numpy.diagonal(beta_phi_t_phi)
        alpha_eigen = old_alpha + eigen
        gamma = (eigen / alpha_eigen).sum()
        # print(f"gamma: {gamma}")

        mn_transpose = numpy.transpose(mn)
        mn_t_mn = numpy.dot(mn_transpose, mn)
        # print(f"mn_t_mn: {mn_t_mn}")

        new_alpha = gamma / mn_t_mn[0][0]

        phi_m = numpy.dot(train, mn)
        denom = numpy.square(numpy.linalg.norm(phi_m - train_target))
        N = train.shape[0]
        new_beta = (N - gamma) / denom
        # print(f"i: {i}, new_alpha: {new_alpha}, beta: {new_beta}")
        alpha_diff = new_alpha - old_alpha
        beta_diff = new_beta - old_beta
        # print(f"alpha_diff: {alpha_diff}, beta_diff: {beta_diff}")
        if alpha_diff < 0.01 and abs(beta_diff) < 0.01:
            break
        old_beta = new_beta
        old_alpha = new_alpha
    lambda_val = new_alpha / new_beta
    return new_alpha, new_beta, lambda_val, mn


def bayesian(train, train_target, test, test_target):
    new_alpha, new_beta, lambda_val, mn = model_selection(train, train_target)
    w = calculate_weight(lambda_val, train, train_target)
    mse = calculate_mse(w, test, test_target)
    return mse, new_alpha, new_beta, lambda_val


def calculate_task1_mse(f_size, train_full, train_target_full, test, test_target, dataset):
    mle_list = []
    bayesian_list = []
    alpha_list = []
    beta_list = []
    lambda_list = []

    for each_freq in f_size:
        train = train_full.head(each_freq)
        train_target = train_target_full.head(each_freq)
        mse_mle = mle(train, train_target, test, test_target, 0)
        bayesian_mse, alpha, beta, lambda_val = bayesian(train, train_target, test, test_target)
        mle_list.append(mse_mle)
        bayesian_list.append(bayesian_mse)
        alpha_list.append(alpha)
        beta_list.append(beta)
        lambda_list.append(lambda_val)

    output = pd.DataFrame({"frequency": f, "size": f_size, "MLE MSE": mle_list, "alpha": alpha_list, "beta": beta_list,
                           "lambda": lambda_list, "Bayesian MSE": bayesian_list})
    print(f"*********** {dataset} Task 1.1 Summary ***********")
    print(output)
    x = numpy.array(f_size)
    y_crime_mle = numpy.array(mle_list)
    y_crime_bayesian = numpy.array(bayesian_list)
    fig, axis = plt.subplots(1, 2, figsize=(15, 5))
    axis[0].plot(x, y_crime_mle, color='b')
    axis[1].plot(x, y_crime_bayesian, color='r')
    axis[0].set(xlabel="Size", ylabel="MSE", title=f"{dataset} MLE MSE")
    axis[0].set_ylim(0, 1)
    axis[0].legend(["MLE MSE"])
    axis[1].set(xlabel="Size", ylabel="MSE", title=f"{dataset} Bayesian MSE")
    axis[1].set_ylim(0, 1)
    axis[1].legend(["Bayesian MSE"])
    plt.savefig(f"Task 1_2 {dataset}.png")


def calculate_task1_3(f_size, train_full, train_target_full, test, test_target, dataset):
    given_lambda_vals = [1, 33, 100, 1000]
    each_lambda_mse = {}
    bayesian_list = []
    alpha_list = []
    beta_list = []
    lambda_list = []
    for each_lambda in given_lambda_vals:
        each_lambda_mse[each_lambda] = []
    for each_freq in f_size:
        train = train_full.head(each_freq)
        train_target = train_target_full.head(each_freq)
        bayesian_mse, alpha, beta, lambda_val = bayesian(train, train_target, test, test_target)
        bayesian_list.append(bayesian_mse)
        alpha_list.append(alpha)
        beta_list.append(beta)
        lambda_list.append(lambda_val)
        for each_lambda in given_lambda_vals:
            mse_mle = mle(train, train_target, test, test_target, each_lambda)
            each_lambda_mse[each_lambda].append(mse_mle)
    output = pd.DataFrame({"Frequency": f, "Size": f_size, "alpha": alpha_list, "beta": beta_list,
                           "lambda": lambda_list, "Bayesian MSE": bayesian_list,
                           "lambda 1": each_lambda_mse[given_lambda_vals[0]],
                           "lambda 33": each_lambda_mse[given_lambda_vals[1]],
                           "lambda 100": each_lambda_mse[given_lambda_vals[2]],
                           "lambda 1000": each_lambda_mse[given_lambda_vals[3]]})
    print(f"\n********** {dataset} Task 1.3 Summary *************")
    print(output)
    x = numpy.array(f_size)
    y_bayesian = numpy.array(bayesian_list)
    y_l1 = numpy.array(each_lambda_mse[given_lambda_vals[0]])
    y_l33 = numpy.array(each_lambda_mse[given_lambda_vals[1]])
    y_l100 = numpy.array(each_lambda_mse[given_lambda_vals[2]])
    y_l1000 = numpy.array(each_lambda_mse[given_lambda_vals[3]])

    fig, axis = plt.subplots(1, 2, figsize=(15, 5))
    axis[0].plot(x, y_bayesian)
    axis[1].plot(x, y_l1)
    axis[1].plot(x, y_l33)
    axis[1].plot(x, y_l100)
    axis[1].plot(x, y_l1000)

    axis[0].set(xlabel="Size", ylabel="MSE", title=f"{dataset} Bayesian")
    axis[0].legend(["Bayesian"])
    axis[0].set_ylim(0, 1)
    axis[1].set(xlabel="Size", ylabel="MSE", title=f"{dataset} NR Linear Regression")
    axis[1].legend(["Lambda-1", "Lambda-33", "Lambda-100", "Lambda-1000"])
    axis[1].set_ylim(0, 1)
    plt.savefig(f"Task 1_3 {dataset}.png")


def task1():
    crime_train_file = "pp2data/train-crime.csv"
    crime_train_full = pd.read_csv(crime_train_file, header=None)
    crime_size = crime_train_full.shape[0]
    crime_train_target_file = "pp2data/trainR-crime.csv"
    crime_train_target_full = pd.read_csv(crime_train_target_file, header=None)
    crime_test_file = "pp2data/test-crime.csv"
    crime_test = pd.read_csv(crime_test_file, header=None)
    crime_test_target_file = "pp2data/testR-crime.csv"
    crime_test_target = pd.read_csv(crime_test_target_file, header=None)

    f_size = [int(crime_size * i) for i in f]
    calculate_task1_mse(
        f_size, crime_train_full, crime_train_target_full, crime_test, crime_test_target, "Crime")
    calculate_task1_3(
        f_size, crime_train_full, crime_train_target_full, crime_test, crime_test_target, "Crime")

    housing_train_file = "pp2data/train-housing.csv"
    housing_train_full = pd.read_csv(housing_train_file, header=None)
    housing_size = housing_train_full.shape[0]
    housing_train_target_file = "pp2data/trainR-housing.csv"
    housing_train_target_full = pd.read_csv(housing_train_target_file, header=None)
    housing_test_file = "pp2data/test-housing.csv"
    housing_test = pd.read_csv(housing_test_file, header=None)
    housing_test_target_file = "pp2data/testR-housing.csv"
    housing_test_target = pd.read_csv(housing_test_target_file, header=None)

    f_size = [int(housing_size * i) for i in f]
    calculate_task1_mse(
        f_size, housing_train_full, housing_train_target_full, housing_test, housing_test_target, "Housing")
    calculate_task1_3(
        f_size, housing_train_full, housing_train_target_full, housing_test, housing_test_target, "Housing")


def cal_log_evidence(train, train_target, test, test_target, M):
    alpha, beta, lambda_val, mn = model_selection(train, train_target)
    phi_t_phi = numpy.dot(numpy.transpose(train), train)
    identity = numpy.identity(phi_t_phi.shape[0])
    alpha_identity = alpha * identity
    beta_phi_t_phi = beta * phi_t_phi
    A = alpha_identity + beta_phi_t_phi
    N = train.shape[0]
    E1 = (beta/2) * numpy.square(numpy.linalg.norm(train_target - numpy.dot(train, mn)))
    E2 = (alpha/2) * numpy.dot(numpy.transpose(mn), mn)
    E = E1 + E2[0][0]
    lev = ((M/2) * math.log(alpha)) + ((N/2) * math.log(beta)) - E - (0.5 * math.log(numpy.linalg.det(A))) - ((N/2) * math.log(2*math.pi))
    mse = calculate_mse(mn, test, test_target)
    return lev, mse, alpha, beta, lambda_val


def calculate_task2(train, train_target, train_one_df, test, test_target, test_one_df, dataset):
    pol_train_df = train_one_df
    pol_test_df = test_one_df
    req_train = train_one_df
    req_test = test_one_df
    mle_list = []
    lev_list = []
    bayesian_list = []
    alpha_list = []
    beta_list = []
    lambda_list = []
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in d:
        pol_train_df = pol_train_df * train
        pol_test_df = pol_test_df * test
        req_train = numpy.concatenate([req_train, pol_train_df], axis=1)
        req_test = numpy.concatenate([req_test, pol_test_df], axis=1)
        mse_mle = mle(req_train, train_target, req_test, test_target, 0)
        mle_list.append(mse_mle)
        lev_mse, lev_bayesian, alpha, beta, lambda_val = cal_log_evidence(req_train, train_target, req_test, test_target, i)
        lev_list.append(lev_mse)
        bayesian_list.append(lev_bayesian)
        alpha_list.append(alpha)
        beta_list.append(beta)
        lambda_list.append(lambda_val)

    output = pd.DataFrame({"Dimension": d, "MLE MSE": mle_list, "alpha": alpha_list, "beta": beta_list,
                           "lambda": lambda_list, "Bayesian MSE": bayesian_list, "Log Evidence": lev_list})
    print(f"\n*********** {dataset} Task 2 Summary ***********")
    print(output)

    x = numpy.array(d)
    y_f3_mle = numpy.array(mle_list)
    y_f3_bayesian = numpy.array(bayesian_list)
    y_f3_lev = numpy.array(lev_list)
    fig, axis = plt.subplots(1, 3, figsize=(25, 8))

    axis[0].plot(x, y_f3_mle, label="F3 MLE")
    axis[1].plot(x, y_f3_bayesian, label="F3 Bayesian")
    axis[2].plot(x, y_f3_lev, label="F3 Log Evidence")

    axis[0].set(xlabel="Size", ylabel="MSE", title=f"{dataset} MLE")
    axis[0].legend(["MLE"])
    axis[0].set_ylim(10000, 1000000)

    axis[1].set(xlabel="Size", ylabel="MSE", title=f"{dataset} Bayesian")
    axis[1].legend(["Bayesian"])
    axis[1].set_ylim(10000, 500000)

    axis[2].set(xlabel="Size", ylabel="Log Evidence", title=f"{dataset} Log Evidence")
    axis[2].legend(["Log Evidence"])
    axis[2].set_ylim(-5000, -1500)
    plt.savefig(f"Task 2 {dataset}.png")


def task2():
    f3_train_file = "pp2data/train-f3.csv"
    f3_train = pd.read_csv(f3_train_file, header=None)
    f3_train_target_file = "pp2data/trainR-f3.csv"
    f3_train_target = pd.read_csv(f3_train_target_file, header=None)
    train_one_data = [1 for x in range(f3_train.shape[0])]
    train_one_df = pd.DataFrame(train_one_data)
    f3_test_file = "pp2data/test-f3.csv"
    f3_test = pd.read_csv(f3_test_file, header=None)
    f3_test_target_file = "pp2data/testR-f3.csv"
    f3_test_target = pd.read_csv(f3_test_target_file, header=None)
    test_one_data = [1 for y in range(f3_test.shape[0])]
    test_one_df = pd.DataFrame(test_one_data)
    calculate_task2(f3_train, f3_train_target, train_one_df, f3_test, f3_test_target, test_one_df, "f3")

    f5_train_file = "pp2data/train-f5.csv"
    f5_train = pd.read_csv(f5_train_file, header=None)
    f5_train_target_file = "pp2data/trainR-f5.csv"
    f5_train_target = pd.read_csv(f5_train_target_file, header=None)
    train_one_f5_data = [1 for x in range(f5_train.shape[0])]
    train_one_f5_df = pd.DataFrame(train_one_f5_data)
    f5_test_file = "pp2data/test-f5.csv"
    f5_test = pd.read_csv(f5_test_file, header=None)
    f5_test_target_file = "pp2data/testR-f5.csv"
    f5_test_target = pd.read_csv(f5_test_target_file, header=None)
    test_one_f5_data = [1 for y in range(f5_test.shape[0])]
    test_one_f5_df = pd.DataFrame(test_one_f5_data)
    calculate_task2(f5_train, f5_train_target, train_one_f5_df, f5_test, f5_test_target, test_one_f5_df, "f5")


task1()
task2()
