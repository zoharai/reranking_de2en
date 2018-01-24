from scipy import stats

import bsbleu
import subprocess
import mteval
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

def calculate_combine_pvalues(old_ranks, new_ranks):
    p_values = []
    for i in range(len(old_ranks)):
        p_values.append(calculate_rank_correlation(old_ranks[i], new_ranks[i])[1])
    return stats.combine_pvalues(p_values)



def calculate_stats(gt_sents, first_sents, score_sents):
    acc_first = len([i for i in range(len(gt_sents)) if gt_sents[i] == first_sents[i]])
    acc_score = len([i for i in range(len(gt_sents)) if gt_sents[i] == score_sents[i]])
    acc_improvment = len([i for i in range(len(gt_sents)) if gt_sents[i] == score_sents[i] and score_sents[i]!= first_sents[i]]  )
    equal_first_score =len([i for i in range(len(score_sents)) if score_sents[i] == first_sents[i]])
    print(acc_first, acc_score,acc_improvment, equal_first_score)

    return [i for i in range(len(score_sents)) if score_sents[i] != first_sents[i]]
# calculate_bleu_from_mteval("newstest2016.tc.en.short", "pipeline_results_f_score")
# calculate_bleu_from_perl("newstest2016.tc.en.short", "newstest2016.de.tc", "pipeline_results_f_score")