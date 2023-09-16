import math
import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate


train_file_loc = "pp1data/training_data.txt"
test_file_loc = "pp1data/test_data.txt"

file_train_list = []
file_test_list = []
with open(train_file_loc) as train:
    for train_line in train:
        for train_word in train_line.split():
            file_train_list.append(train_word)

with open(test_file_loc) as test:
    for test_line in test:
        for test_word in test_line.split():
            file_test_list.append(test_word)

N = len(file_train_list)
file_train_size = N
file_test_size = len(file_test_list)

all_sizes = [N / 128, N / 64, N / 16, N / 4, N]


def train_perplexity(sizes, train_list):
    train_final_results = []
    train_max_mle_perplexity = 0

    train_mle_perplexity_list = []
    train_map_perplexity_list = []
    train_pd_perplexity_list = []

    for size in sizes:
        freq_dict = {}
        for i in range(int(size)):
            if train_list[i] in freq_dict:
                freq_dict[train_list[i]] += 1
            else:
                freq_dict[train_list[i]] = 1
        unique_word_count = len(freq_dict)

        mle_prob_dict = {}
        for each_word in freq_dict:
            mle_prob_dict[each_word] = float((freq_dict[each_word]) / size)

        map_prob_dict = {}
        ak = 2
        K = unique_word_count
        a0 = K * ak
        for each_word in freq_dict:
            map_prob_dict[each_word] = float((freq_dict[each_word] + ak - 1) / (size + a0 - K))

        pd_prob_dict = {}
        for each_word in freq_dict:
            pd_prob_dict[each_word] = float((freq_dict[each_word] + ak) / (size + a0))

        train_mle_perplexity_summation = 0
        train_map_perplexity_summation = 0
        train_pd_perplexity_summation = 0

        for each_word in freq_dict:
            if mle_prob_dict[each_word] != 0:
                train_mle_perplexity_summation += float(math.log(mle_prob_dict[each_word]))
            if map_prob_dict[each_word] != 0:
                train_map_perplexity_summation += float(math.log(map_prob_dict[each_word]))
            if pd_prob_dict[each_word] != 0:
                train_pd_perplexity_summation += float(math.log(pd_prob_dict[each_word]))

        train_mle_perplexity_exp_val = (train_mle_perplexity_summation / size) * (-1)
        train_mle_perplexity = math.exp(train_mle_perplexity_exp_val)
        if train_mle_perplexity > train_max_mle_perplexity:
            train_max_mle_perplexity = train_mle_perplexity
        print(f"Perplexity of train data of size {size} using MLE model: {train_mle_perplexity}")
        train_mle_perplexity_list.append(train_mle_perplexity)

        train_map_perplexity_exp_val = (train_map_perplexity_summation / size) * (-1)
        train_map_perplexity = math.exp(train_map_perplexity_exp_val)
        print(f"Perplexity of train data of size {size} using MAP model: {train_map_perplexity}")
        train_map_perplexity_list.append(train_map_perplexity)

        train_pd_perplexity_exp_val = (train_pd_perplexity_summation / size) * (-1)
        train_pd_perplexity = math.exp(train_pd_perplexity_exp_val)
        print(f"Perplexity of train data of size {size} using predictive distribution model: {train_pd_perplexity}")
        train_pd_perplexity_list.append(train_pd_perplexity)
        train_final_results.append([train_mle_perplexity, train_map_perplexity, train_pd_perplexity])
        print("\n\n")

    print("************** Train Data ****************")
    print(tabulate(train_final_results, headers=["MLE", "MAP", "Predictive Distribution"]))

    x = np.array(sizes)
    y_mle = np.array(train_mle_perplexity_list)
    y_map = np.array(train_map_perplexity_list)
    y_pd = np.array(train_pd_perplexity_list)

    fig, axis = plt.subplots(2, 1, figsize=(15, 15))
    axis[0].plot(x, y_mle, label='MLE Perplexity')
    axis[0].plot(x, y_map, label='MAP Perplexity')
    axis[0].plot(x, y_pd, label='PD Perplexity')
    axis[0].set(xlabel="Size N", ylabel="Perplexity", title="Model Perplexity plot for Train Data")

    X_axis = np.arange(len(x))
    axis[1].bar(X_axis - 0.1, y_mle, 0.1, label='MLE Perplexity')
    axis[1].bar(X_axis + 0.0, y_map, 0.1, label='MAP Perplexity')
    axis[1].bar(X_axis + 0.1, y_pd, 0.1, label='PD Perplexity')
    axis[1].set(xlabel="Size N", ylabel="Perplexity", title="Bar Graph for Model Perplexity of Train Data")
    plt.figlegend()
    plt.savefig("task1_train_perplexity_plots.png")
    plt.show()

    return train_final_results


def test_perplexity(sizes, train_list, test_list, test_size):
    test_mle_perplexity_list = []
    test_map_perplexity_list = []
    test_pd_perplexity_list = []
    test_final_results = []
    max_mle_perplexity = 0
    for size in sizes:
        freq_dict = {}
        for i in range(int(size)):
            if train_list[i] in freq_dict:
                freq_dict[train_list[i]] += 1
            else:
                freq_dict[train_list[i]] = 1
        unique_word_count = len(freq_dict)

        for each_word in test_list:
            if each_word not in freq_dict:
                freq_dict[each_word] = 0

        mle_prob_dict = {}
        for each_word in freq_dict:
            mle_prob_dict[each_word] = float((freq_dict[each_word]) / size)

        map_prob_dict = {}
        ak = 2
        K = unique_word_count
        a0 = K * ak
        for each_word in freq_dict:
            map_prob_dict[each_word] = float((freq_dict[each_word] + ak - 1) / (size + a0 - K))

        pd_prob_dict = {}
        for each_word in freq_dict:
            pd_prob_dict[each_word] = float((freq_dict[each_word] + ak) / (size + a0))
        test_mle_perplexity_summation = 0
        is_test_mle_infinity = False
        test_map_perplexity_summation = 0
        test_pd_perplexity_summation = 0

        for each_word in test_list:
            if mle_prob_dict[each_word] != 0:
                test_mle_perplexity_summation += float(math.log(mle_prob_dict[each_word]))
            else:
                is_test_mle_infinity = True
            if map_prob_dict[each_word] != 0:
                test_map_perplexity_summation += float(math.log(map_prob_dict[each_word]))
            if pd_prob_dict[each_word] != 0:
                test_pd_perplexity_summation += float(math.log(pd_prob_dict[each_word]))

        if not is_test_mle_infinity:
            test_mle_perplexity_exp_val = (test_mle_perplexity_summation / test_size) * (-1)
            test_mle_perplexity = math.exp(test_mle_perplexity_exp_val)
            if test_mle_perplexity > max_mle_perplexity:
                max_mle_perplexity = test_mle_perplexity
            print(f"\n\nPerplexity of test data of size {size} using MLE model: {test_mle_perplexity}")
        else:
            test_mle_perplexity = "infinity"
            print(f"\n\nPerplexity of test data of size {size} using MLE model: infinity")
        test_mle_perplexity_list.append(test_mle_perplexity)

        test_map_perplexity_exp_val = (test_map_perplexity_summation / test_size) * (-1)
        test_map_perplexity = math.exp(test_map_perplexity_exp_val)
        print(f"Perplexity of test data of size {size} using MAP model: {test_map_perplexity}")
        test_map_perplexity_list.append(test_map_perplexity)

        test_pd_perplexity_exp_val = (test_pd_perplexity_summation / test_size) * (-1)
        test_pd_perplexity = math.exp(test_pd_perplexity_exp_val)
        print(f"Perplexity of test data of size {size} using predictive distribution model: {test_pd_perplexity}")
        test_pd_perplexity_list.append(test_pd_perplexity)
        test_final_results.append([test_mle_perplexity, test_map_perplexity, test_pd_perplexity])
        print("\n\n")

    if "infinity" in test_mle_perplexity_list:
        infinity_mle = (max_mle_perplexity + 1000)
        test_mle_perplexity_list = list(map(lambda x: infinity_mle if x == "infinity" else x, test_mle_perplexity_list))

    print("************** Test Data ****************")
    print(tabulate(test_final_results, headers=["MLE", "MAP", "Predictive Distribution"]))

    x = np.array(sizes)
    y_mle = np.array(test_mle_perplexity_list)
    y_map = np.array(test_map_perplexity_list)
    y_pd = np.array(test_pd_perplexity_list)

    fig, axis = plt.subplots(2, 1, figsize=(15, 15))
    axis[0].plot(x, y_mle, color='b', label='MLE Perplexity')
    axis[0].plot(x, y_map, color='r', label='MAP Perplexity')
    axis[0].plot(x, y_pd, color='g', label='PD Perplexity')
    axis[0].set(xlabel="Size N", ylabel="Perplexity", title="Model Perplexity plot for Test Data")

    X_axis = np.arange(len(x))
    axis[1].bar(X_axis - 0.1, y_mle, 0.1, label='MLE Perplexity')
    axis[1].bar(X_axis + 0.0, y_map, 0.1, label='MAP Perplexity')
    axis[1].bar(X_axis + 0.1, y_pd, 0.1, label='PD Perplexity')
    axis[1].set(xlabel="Size N", ylabel="Perplexity", title="Bar Graph for Model Perplexity of Test Data")
    plt.figlegend()
    plt.savefig("task1_test_perplexity_plots.png")
    plt.show()
    return test_final_results


train_final_results = train_perplexity(all_sizes, file_train_list)
test_final_results = test_perplexity(all_sizes, file_train_list, file_test_list, file_test_size)

