import json
import glob
import random
import heapq
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


class PP4:
    def __init__(self):
        ""

    def gibbs_lda(self, path):
        start = datetime.now()
        k = 2
        n_iters = 500
        alpha = 5 / k
        beta = 0.01
        doc_ind_max = 0
        doc_range = {}
        all_words = []
        doc_index = 0
        index_doc_map = {}
        sum_docs_topics = {}

        d = []
        for file in glob.glob(f"{path}*"):
            if not file.endswith('.csv'):
                with open(file) as f:
                    for each_line in f:
                        line_words = each_line.split()
                        all_words.extend(line_words)
                        d.extend([doc_index for each in line_words])
                index_doc_map[doc_index] = file
                doc_range[doc_index] = [doc_ind_max, doc_ind_max + len(line_words)]
                doc_ind_max = doc_ind_max + len(line_words)
                sum_docs_topics[doc_index] = 0

                doc_index += 1

        n_words = len(all_words)
        unique_words = list(set(all_words))
        word_index = 0
        index_word_map = dict()
        word_index_map = dict()
        for each_word in unique_words:
            index_word_map[word_index] = each_word
            word_index_map[each_word] = word_index
            word_index += 1

        w = [word_index_map[each_word] for each_word in all_words]
        k_list = [i for i in range(k)]
        random.seed(100)
        z = [random.choice(k_list) for i in range(n_words)]
        v = len(unique_words)
        pi = [i for i in range(n_words)]
        random.shuffle(pi)
        C_d = [[0 for each_k in range(k)] for each_doc in doc_range]
        C_t = [[0 for each_word in unique_words] for each_topic in range(k)]
        words_bag = [[0 for each_word in unique_words] for each_doc in doc_range]

        for each_doc_ind, each_range in doc_range.items():
            doc_z = z[each_range[0]:each_range[1]]
            for each_k in doc_z:
                C_d[each_doc_ind][each_k] += 1
                sum_docs_topics[each_doc_ind] += 1

        for i in range(n_words):
            C_t[z[i]][word_index_map[all_words[i]]] += 1

        for each_doc_ind, each_range in doc_range.items():
            doc_words = all_words[each_range[0]:each_range[1]]
            for each_word in doc_words:
                words_bag[each_doc_ind][word_index_map[each_word]] += 1

        P = [0 for i in range(k)]
        for i in range(n_iters):
            for n in range(n_words):
                word = w[pi[n]]
                topic = z[pi[n]]
                doc = d[pi[n]]
                C_d[doc][topic] -= 1
                C_t[topic][word] -= 1
                for each_k in range(k):
                    sum_ctk = sum(C_t[each_k])
                    P[each_k] = ((C_t[each_k][word] + beta) / ((v * beta) + sum_ctk)) * (
                            (C_d[doc][each_k] + alpha) / ((k * alpha) + sum_docs_topics[doc]))
                p_sum = sum(P)
                P = [each_p / p_sum for each_p in P]
                topic = P.index(max(P))
                z[pi[n]] = topic
                C_d[doc][topic] += 1
                C_t[topic][word] += 1

        topic_max_words = []

        for i in range(len(C_t)):
            max_5 = heapq.nlargest(3, [(C_t[i][j], j) for j in range(len(C_t[i]))])
            topic_words = []
            for each_ind in max_5:
                topic_words.append(index_word_map[each_ind[1]])
            topic_max_words.append(",".join(topic_words))

        with open("topicwords.csv", "w") as f:
            for each in topic_max_words:
                f.write(f"{each}\n")

        print(f"unique words : {len(unique_words)}")
        print(f"Documents : {doc_index}")
        execution_time = datetime.now() - start
        print(f"Execution Time: {execution_time}")
        topic_rep = []
        for each_doc in C_d:
            sum_topics = sum(each_doc)
            k_values = []
            for each_topic in each_doc:
                k_values.append((each_topic + alpha) / ((k * alpha) + sum_topics))
            topic_rep.append(k_values)

        with open("bag_of_words", "w") as w:
            w.write(str(words_bag))
        with open("topic_rep", "w") as w:
            w.write(str(topic_rep))

        return topic_rep, words_bag

    def cal_gen_error_rate(self, train_full, labels_train_full, dt_train_size, test, labels_test):
        run_count = 30
        runs = [i for i in range(run_count)]
        mean_error_rate = []
        sd_error_rate = []
        for each_train_size in all_train_sizes:
            sum_error_rate = 0
            error_rates = []
            for each_run in runs:
                train_size = round(each_train_size * dt_train_size)
                rows = random.sample([i for i in range(dt_train_size)], train_size)

                train = train_full.iloc[rows]
                train_labels = labels_train_full.iloc[rows]
                m1_sum_tx = 0
                m2_sum_tx = 0
                N1 = train_labels[1].sum()
                N2 = train_size - N1
                N = N1 + N2
                train = train.reset_index()
                for index, each_eg in train.iterrows():
                    each_eg = each_eg[1:]
                    tx = train_labels.iloc[index][1]
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
                    tx = train_labels.iloc[index][1]
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
                    except:
                        val = 0 if each < 0 else 1
                    P.append(val)
                    cal_label.append(1 if val >= 0.5 else 0)
                error_sum = 0
                for i in range(len(cal_label)):
                    error_sum += 1 if cal_label[i] != labels_test.iloc[i][1] else 0
                error_rate = error_sum / len(cal_label)
                sum_error_rate += error_rate
                error_rates.append(error_rate)
            mean_error_rate.append(np.mean(error_rates))
            sd_error_rate.append(np.std(error_rates))
        result = pd.DataFrame({"Train Size": all_train_sizes, "Mean Error Rate": mean_error_rate,
                               "SD error rate": sd_error_rate})
        print(result)
        return mean_error_rate, sd_error_rate

    def cal_task2(self, topic_rep, bag_words):
        topic_rep_data = pd.DataFrame(topic_rep)
        labels_data = pd.read_csv(f'{path}index.csv', header=None, index_col=0)
        topic_rep_size = topic_rep_data.shape[0]
        topic_rep_train_size = round(2 * topic_rep_size / 3)
        topic_rep_test_size = round(1 * topic_rep_size / 3)
        topic_rep_train_full = topic_rep_data.head(topic_rep_train_size)
        topic_rep_labels_train_full = labels_data.head(topic_rep_train_size)
        topic_rep_test = topic_rep_data.tail(topic_rep_test_size)
        labels_test = labels_data.tail(topic_rep_test_size)
        print("*********************Error rates for the LDA***********************")
        tr_mean, tr_std = self.cal_gen_error_rate(topic_rep_train_full, topic_rep_labels_train_full,
                                                  topic_rep_train_size, topic_rep_test, labels_test)

        bw_data = pd.DataFrame(bag_words)
        bw_size = bw_data.shape[0]
        bw_train_size = round(2 * bw_size / 3)
        bw_test_size = round(1 * bw_size / 3)
        bw_train_full = bw_data.tail(bw_train_size)
        bw_labels_train_full = labels_data.tail(bw_train_size)
        bw_test = bw_data.head(bw_test_size)
        print("*********************Error rates for Bag of words***********************")
        bw_mean, bw_std = self.cal_gen_error_rate(bw_train_full, bw_labels_train_full, bw_train_size, bw_test,
                                                  labels_test)

        # fig, axis = plt.subplots(1, 2, figsize=(15, 5))
        x = np.array(all_train_sizes)
        yt_mean = np.array(tr_mean)
        yt_sd = np.array(tr_std)

        yb_mean = np.array(bw_mean)
        yb_sd = np.array(bw_std)

        plt.plot(x, yt_mean, color="red", label="LDA Error")
        plt.errorbar(x, yt_mean, yerr=yt_sd, fmt='o')
        plt.plot(x, yb_mean, color="blue", label="Bag of words Error")
        plt.errorbar(x, yb_mean, yerr=yb_sd, fmt='o')
        plt.xlabel("Size")
        plt.ylabel("Test Error")
        plt.legend()
        plt.savefig("plot.jpg")


path = "pp4data/artificial/"
all_train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
obj = PP4()
topic_rep, bag_words = obj.gibbs_lda(path)
# obj.gibbs_lda(path)
# top = open("topic_rep", "r")
# top_content = top.read()
# topic_rep = json.loads(top_content)
# bag = open("bag_of_words", "r")
# bag_content = bag.read()
# bag_words = json.loads(bag_content)
# obj.cal_task2(topic_rep, bag_words)
