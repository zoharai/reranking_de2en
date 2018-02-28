from scipy import stats

import bsbleu
import math
import subprocess
import mteval
import random
from multiprocessing import Pool

def calculate_bleu_score(R,C, files=False):
    return bsbleu.bleu(R,C, False, files=files, bootstrap=1000)


def calculate_bleu_from_perl(reference, german, candidate):
    subprocess.call(["perl", "mteval-v14.pl", reference, german, candidate])

def calculate_bleu_from_mteval(reference, candidate):
    scorer = mteval.MTEvalV13aBLEUScorer()
    score = scorer.compute(reference, candidate)
    print(score)


def calculate_rank_correlation(old_rank, new_rank):
    return stats.spearmanr(old_rank, new_rank)



def calculate_permutation(old_ranks, new_ranks):
        mean = []
        for i in range(len(old_ranks)):
            permutation = random.sample(old_ranks[i], len(old_ranks[i]))
            rho, p_value = calculate_rank_correlation(permutation, new_ranks[i])
            if not math.isnan(p_value):
                mean.append(rho)
        return sum(mean) / len(mean)

def calculate_combine_pvalues(alignments_dict):
    old_ranks = []
    new_ranks = []
    count = 0
    for _, scores in alignments_dict.items():
        old_rank = []
        new_rank = []
        for score in sorted(scores, reverse=True):
            for a in scores[score]:
                # if len(old_ranks) != 0 :
                #     en_index = a._en_index-len(old_ranks[-1])*len(old_ranks)
                old_rank.append( a._en_prob)
                new_rank.append(score)
        count += len(old_rank)
        old_ranks.append(old_rank)
        new_ranks.append(new_rank)
    #
    N = 100
    means = []
    with Pool(processes=5) as pool:
        means = pool.starmap(calculate_permutation, [(old_ranks,new_ranks)]*N)

    # N = 10000
    # means = []
    # for _ in range(N):
    #     mean = []
    #     for i in range(len(old_ranks)):
    #         permutation = random.sample(old_ranks[i], len(old_ranks[i]))
    #         rho, p_value = calculate_rank_correlation(permutation, new_ranks[i])
    #         if not math.isnan(p_value):
    #             mean.append(rho)
    #     means.append(sum(mean)/len(mean))


    print (means)
    p_values = []
    rhos = []
    for i in range(len(old_ranks)):
        rho, p_value = calculate_rank_correlation(old_ranks[i], new_ranks[i])
        if not math.isnan(p_value):
            p_values.append(p_value)
            rhos.append(rho)
    real_mean = sum(rhos)/len(rhos)
    p_value = len([x for x in means if abs(x)>= abs(real_mean)])/N
    return p_value,real_mean



def calculate_stats(gt_sents, first_sents, score_sents):
    acc_first = len([i for i in range(len(gt_sents)) if gt_sents[i] == first_sents[i]])
    acc_score = len([i for i in range(len(gt_sents)) if gt_sents[i] == score_sents[i]])
    acc_improvment = len([i for i in range(len(gt_sents)) if gt_sents[i] == score_sents[i] and score_sents[i]!= first_sents[i]]  )
    equal_first_score =len([i for i in range(len(score_sents)) if score_sents[i] == first_sents[i]])
    print(acc_first, acc_score,acc_improvment, equal_first_score)

    return [i for i in range(len(score_sents)) if score_sents[i] != first_sents[i]]
# calculate_bleu_from_mteval("newstest2016.tc.en.short", "pipeline_results_f_score")
# calculate_bleu_from_perl("newstest2016.tc.en.short", "newstest2016.de.tc", "pipeline_results_f_score")