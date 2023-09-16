import math
import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate

train_file = "pp1data/training_data.txt"
test_file = "pp1data/test_data.txt"


train_list = []
test_list = []
with open(train_file) as train:
    for train_line in train:
        for train_word in train_line.split():
            train_list.append(train_word)

with open(test_file) as test:
    for test_line in test:
        for test_word in test_line.split():
            test_list.append(test_word)

N = len(train_list)
test_size = len(test_list)

size = int(N / 128)
list_ak = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
freq_dict = {}
unique_train_list = []
for i in range(int(size)):
    if train_list[i] in freq_dict:
        freq_dict[train_list[i]] += 1
    else:
        unique_train_list.append(train_list[i])
        freq_dict[train_list[i]] = 1

unique_word_count = len(freq_dict)
K = unique_word_count
for each_word in test_list:
    if each_word not in freq_dict:
        freq_dict[each_word] = 0

final_result = []
list_log_evidence = []
list_pd = []
for each_ak in list_ak:
    ak = each_ak
    a0 = K * ak

    gamma_a0 = math.factorial(a0 - 1)
    log_gamma_a0 = math.log(gamma_a0)

    log_gamma_ak_mk = 0
    for each_word in unique_train_list:
        mk = freq_dict[each_word]
        gamma_ak_mk = math.factorial(ak + mk - 1)
        log_gamma_ak_mk += gamma_ak_mk
    gamma_a0_n = math.factorial(a0 + size - 1)
    log_gamma_a0_n = math.log(gamma_a0_n)

    gamma_ak = math.factorial(ak - 1)
    log_gamma_ak = K * gamma_ak

    log_evidence = log_gamma_a0 + log_gamma_ak_mk - log_gamma_a0_n - log_gamma_ak
    list_log_evidence.append(log_evidence)

    pd_prob_dict = {}
    for each_word in freq_dict:
        pd_prob_dict[each_word] = float((freq_dict[each_word] + ak) / (size + a0))

    pd_perplexity_summation = 0

    for each_word in test_list:
        pd_perplexity_summation += float(math.log(pd_prob_dict[each_word]))

    pd_perplexity_exp_val = (pd_perplexity_summation / test_size) * (-1)
    pd_perplexity = math.exp(pd_perplexity_exp_val)
    list_pd.append(pd_perplexity)

    final_result.append([ak, log_evidence, pd_perplexity])

print(tabulate(final_result, headers=["alpha_k", "Log Evidence", "Predictive Distribution"]))

x = np.array(list_ak)
y_log_evidence = np.array(list_log_evidence)
y_pd_pp = np.array(list_pd)

fig, axis = plt.subplots(3, 1, figsize=(18, 25))
axis[0].plot(x, y_log_evidence, color='b', label='Log Evidence')
axis[0].plot(x, y_pd_pp, color='r', label='PD Perplexity')

axis[0].set(xlabel="alpha_k", ylabel="Log Evidence/Perplexity", title="Plotting both Log Evidence/Perplexity vs alpha_k")

axis[1].plot(x, y_log_evidence, color='b', label='Log Evidence')
axis[1].set(xlabel="alpha_k", ylabel="Log Evidence", title="alpha_k vs Log Evidence")

axis[2].plot(x, y_pd_pp, color='r', label='PD Perplexity')
axis[2].set(xlabel="alpha_k", ylabel="Perplexity", title="alpha_k vs Perplexity")

plt.savefig("task2_model_selection_plots.png")
plt.show()
