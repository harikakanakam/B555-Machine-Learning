import math


file_121 = "pp1data/pg121.txt.clean"
file_141 = "pp1data/pg141.txt.clean"
file_1400 = "pp1data/pg1400.txt.clean"

list_121 = []
list_141 = []
list_1400 = []
with open(file_121) as f121:
    for each_line in f121:
        for each_word in each_line.split():
            list_121.append(each_word)

with open(file_141) as f141:
    for each_line in f141:
        for each_word in each_line.split():
            list_141.append(each_word)

with open(file_1400) as f1400:
    for each_line in f1400:
        for each_word in each_line.split():
            list_1400.append(each_word)
N = len(list_121)
size = N
freq_dict = {}

for i in range(int(size)):
    if list_121[i] in freq_dict:
        freq_dict[list_121[i]] += 1
    else:
        freq_dict[list_121[i]] = 1

unique_word_count = len(freq_dict)

for each_word in list_141:
    if each_word not in freq_dict:
        freq_dict[each_word] = 0
for each_word in list_1400:
    if each_word not in freq_dict:
        freq_dict[each_word] = 0


ak = 1
K = unique_word_count
a0 = K*ak

pd_prob_dict = {}
for each_word in freq_dict:
    pd_prob_dict[each_word] = float((freq_dict[each_word]+ak)/(size+a0))


pd_121_perplexity_summation = 0
for each_word in list_121:
    pd_121_perplexity_summation += math.log(pd_prob_dict[each_word])

pd_121_perplexity_exp_val = (pd_121_perplexity_summation / len(list_121)) * (-1)
pd_121_perplexity = math.exp(pd_121_perplexity_exp_val)

pd_141_perplexity_summation = 0
for each_word in list_141:
    if pd_prob_dict[each_word] != 0:
        pd_141_perplexity_summation += float(math.log(pd_prob_dict[each_word]))

pd_141_perplexity_exp_val = (pd_141_perplexity_summation / len(list_141)) * (-1)
pd_141_perplexity = math.exp(pd_141_perplexity_exp_val)

pd_1400_perplexity_summation = 0
for each_word in list_1400:
    if pd_prob_dict[each_word] != 0:
        pd_1400_perplexity_summation += float(math.log(pd_prob_dict[each_word]))

pd_1400_perplexity_exp_val = (pd_1400_perplexity_summation / len(list_1400)) * (-1)
pd_1400_perplexity = math.exp(pd_1400_perplexity_exp_val)

print(pd_141_perplexity)
print(pd_1400_perplexity)
