# End to end pipeline for reranking traslated de-en sentences.
# author - Zohar Aizenbud

import spacy
from comparison_v2 import Alignment
import pipeline_stats
import pickle
import operator
from forAligner import add_spaces_to_hypen
import sys

class ProgressBar(object):
    def __init__(self, iterations):
        self.chars = 80
        self.ticks = 0
        self.gap = iterations / self.chars
        print("+" + (self.chars-2) * "-" + "+")

    def tick(self):
        self.ticks += 1
        prev_stat = int((self.ticks-1) / self.gap)
        curr_stat = int(self.ticks / self.gap)
        if prev_stat != curr_stat:
            sys.stdout.write("#")
            sys.stdout.flush()



def create_tree_for_latex(args, sent):
    tree = "\scalebox{0.7}{\n\\begin{dependency}[theme = simple]\n\\begin{deptext}\n"
    tree+= "\&".join(sent.split())
    tree+= "\\\\  \n\end{deptext}\n"
    for i, tup in args.items():
        if tup[1] == "ROOT":
            tree+= "\deproot{"+str(i+1)+"}{ROOT}\n"
        else:
            if tup[1] != "punct":
                tree += " \depedge{"+str(tup[0]+1)+"}{"+str(i+1)+"}{"+tup[1]+"}\n"
    tree += "\n\end{dependency}\n}\n\\\\"
    return tree


def fix_de_args(de_doc):
    de_args = {}
    for token in de_doc:
        if token.head != token and (token.head.tag_.startswith("VA") or token.head.tag_.startswith("VM")):
            if token.pos_ == "VERB" or token.dep_ == "pd":
                de_args[token.head.i] = (token.i, "cop||aux")
                if token.head.dep_ == "ROOT":
                    de_args[token.i] = (token.i, "ROOT")
                else:
                    de_args[token.i] = (token.head.head.i, token.head.dep_)

                for child in token.head.children:
                    if child != token:
                        de_args[child.i] = (token.i, child.dep_)



    for token in de_doc:
        if "mo" in token.dep_.split("||") and token.pos_ != "ADV" and (token.head.pos_== "VERB" or token.head.i in de_args):
            if token.pos_ == "ADP": #preposition
                for child in token.children:
                    if child.dep_ == "nk":
                        if token.head.i in de_args and de_args[token.head.i][1] == "cop||aux":
                            de_args[child.i] = (de_args[token.head.i][0], "pverb")
                            de_args[token.i] = (de_args[token.head.i][0], "prep")
                        else:
                            de_args[child.i] = (token.head.i, "pverb")
                            de_args[token.i] = (token.head.i, "prep")
                        break
            else:
                if token.head.i in de_args and de_args[token.head.i][1] == "cop||aux":
                    de_args[token.i] = (de_args[token.head.i][0], "verbmo")
                else:
                    de_args[token.i] = (token.head.i, "verbmo")

    for token in de_doc:
        if token.dep_ == "op" and token.head.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "nk":
                    if token.head.i in de_args and de_args[token.head.i][1] == "cop||aux":
                        de_args[child.i] = (de_args[token.head.i][0], "pverb")
                        de_args[token.i] = (de_args[token.head.i][0], "prep")
                    else:
                        de_args[child.i] = (token.head.i, "pverb")
                        de_args[token.i] = (token.head.i, "prep")
                    break

    for token in de_doc:
        if token.i not in de_args:
            de_args[token.i] = (token.head.i, token.dep_)

    tree = create_tree_for_latex(de_args, de_doc.string)
    print(tree)
    return de_args

def fix_en_args(en_doc):
    en_args = {}

    for token in en_doc:
        if token.head.lemma_ == "be" and token.dep_ in ["acomp", "amod", "xcomp", "attr"] :
            if token.head.dep_ == "ROOT":
                en_args[token.i] = (token.i, "ROOT")
            else:
                en_args[token.i] = (token.head.head.i, token.head.dep_)

            en_args[token.head.i] = (token.i, "cop")
            for child in token.head.children:
                if child != token:
                    en_args[child.i] = (token.i, child.dep_)

    for token in en_doc:
        if token.dep_ in ["pcomp", "pobj"] and (token.head.head.pos_ == "VERB" or (token.head.head.dep_ in ["acomp", "amod", "xcomp", "attr"] and token.head.head.head.lemma_ == "be")):
            if token.head.head.lemma_ == "be" and token.head.head.i in en_args:
                en_args[token.i] = (en_args[token.head.head.i][0], "pverb")
                en_args[token.head.i] = (en_args[token.head.head.i][0], "prep")
            else:
                en_args[token.i] = (token.head.head.i, "pverb")
                en_args[token.head.i] = (token.head.head.i, "prep")




    for token in en_doc:
        if token.i not in en_args:
            en_args[token.i] = (token.head.i, token.dep_)

    tree = create_tree_for_latex(en_args, en_doc.string)
    print(tree)
    return en_args


def build(alignment_file, outputfile, add_gt=False, gt_file = None, add_prob=True):
    alignments = []
    alignments_dict = {}
    en_model = spacy.load("en")
    de_model = spacy.load("de")

    with open("newstest2016.en.output.probability.long.nbest", "r") as f:
        prob_lines = f.readlines()

    #build alignments objects
    with open(alignment_file, "r") as f:
        prev_de_sent = ""
        de_args = {}
        j = -1
        lines = f.readlines()
        p = ProgressBar(len(lines))
        for i, line in enumerate(lines):
            #print(i)
            de, en, align, score = line.split("|||")
            if de != prev_de_sent:
                de_doc = de_model(de.strip())
                de_args = fix_de_args(de_doc)

                j+=1
                prev_de_sent = de

            en_doc = en_model(en.strip())
            en_args =  fix_en_args(en_doc)
            prob = 0
            if add_prob:
                prob = float(prob_lines[i].strip())
            a = Alignment(en.strip(),de.strip(),align, i, j,en_args ,de_args, prob)
            # a.comparison()
            alignments.append(a)
            alignments_dict[(j,i)] = a
            p.tick()

    if add_gt:
        # add gt english sentences
        with open(gt_file) as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                for j in range(int(i) * 100, (int(i) + 1) * 100):
                    if (int(i), j) in alignments_dict:
                        gt_sent = add_spaces_to_hypen(line.strip().replace("``", '"').replace("''" ,'"'), with_hypen=True)
                        alignments_dict[(int(i), j)].update_gt_en_sent(gt_sent)


    # save to pickle:
    with open(outputfile, 'wb') as output:
        for a in alignments:
            pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

    return alignments, alignments_dict


def organize_results(all_unique_indices):
    with open("pipeline_results_all_long", "w") as output:
        with open("pipeline_results_long_gt_scores", "r") as f1:
            with open("pipeline_results_long_take_the_first", "r") as f2:
                with open("pipeline_results_long_precision", "r") as f3:
                    with open("pipeline_results_long_recall", "r") as f4:
                        with open("pipeline_results_long_f_score", "r") as f5:
                            with open("newstest2016.de.tc", "r") as f6:
                                gt = f1.readlines()
                                first = f2.readlines()
                                precision = f3.readlines()
                                recall = f4.readlines()
                                fscore = f5.readlines()
                                german = f6.readlines()
                                for i in sorted(all_unique_indices):
                                    output.write("i:" + str(i)+"\n")
                                    output.write("german:\n" + german[i])
                                    output.write("gt:\n" + gt[i])
                                    output.write("first:\n" + first[i])
                                    if precision[i].split('\t')[0] != first[i].split('\t')[0]:
                                        output.write("precision:\n" + precision[i])
                                    if recall[i].split('\t')[0] != first[i].split('\t')[0]:
                                        output.write("recall:\n" + recall[i])
                                    if fscore[i].split('\t')[0] != first[i].split('\t')[0]:
                                        output.write("f score:\n" + fscore[i])
                                    output.write("***************\n")


def build_project():
    alignments, alignments_dict = build('test_alignments_long_NO_DUPLICATES', 'alignments_data_without_duplicates.pkl')
    gt, gt_dict = build('test_alignments_gt', 'alignments_data_gt.pkl', add_prob=False)
    print( len(alignments), len(gt))




def run_pipeline():
    #len(alignments) = 24281
    #len(gt) =

    ALIGNMENT_LEN = 299400
    NO_DUPLICATES_ALIGNMENTS_LEN = 254134
    GT_LEN = 2994
    gt_alignments = []
    gt_scores = []
    gt_sents = []
    all_unique_indices = []
    by_precision_de_dict = {}
    by_recall_de_dict = {}
    by_fscore_de_dict = {}
    by_first_de_dict = {}
    without_predicates = []

    with open('alignments_data_gt.pkl', 'rb') as input:
        for i in range(GT_LEN):
            a = pickle.load(input)
            a.comparison()
            gt_alignments.append(a)
            gt_scores.append((a._en_sent,str(a._precision), str(a._recall), str(a._f_score)))
            gt_sents.append(a._en_sent)

    with open("pipeline_results_long_gt_scores", "w") as f:
       f.write("\n".join(["\t".join(tup) for tup in gt_scores]))

    with open("pipeline_results_long_gt", "w") as f:
       f.write("\n".join([tup[0] for tup in gt_scores]))

    # with open('alignments_data_long.pkl', 'rb') as input:
    #    for i in range(ALIGNMENT_LEN):
    with open('alignments_data_without_duplicates.pkl', 'rb') as input:
       for i in range(NO_DUPLICATES_ALIGNMENTS_LEN):
            # print(i)
            a = pickle.load(input)
            a.comparison()

            #filter out sentences without predicates
            if a._without_predicate == True:
                without_predicates.append(a)
                continue



            if a._de_index not in by_precision_de_dict:
                by_precision_de_dict[a._de_index] = {}
                by_recall_de_dict[a._de_index] = {}
                by_fscore_de_dict[a._de_index] = {}
                by_first_de_dict[a._de_index] = a

            if a._precision not in by_precision_de_dict[a._de_index]:
                by_precision_de_dict[a._de_index][a._precision] = []

            if a._recall not in by_recall_de_dict[a._de_index]:
                by_recall_de_dict[a._de_index][a._recall] = []

            if a._f_score not in by_fscore_de_dict[a._de_index]:
                by_fscore_de_dict[a._de_index][a._f_score] = []

            by_precision_de_dict[a._de_index][a._precision].append(a)
            by_recall_de_dict[a._de_index][a._recall].append(a)
            by_fscore_de_dict[a._de_index][a._f_score].append(a)



    print ("len of filtered out sentences: " + str(len(set([a._de_index for a in without_predicates]))))

    with open("pipeline_results_long_gt_filtered", "w") as f:
        f.write("\n".join([tup[0] for i, tup in enumerate(gt_scores) if i not in [a._de_index for a in without_predicates]]))

    p_value, real_mean = pipeline_stats.calculate_combine_pvalues(by_precision_de_dict)
    print(p_value,real_mean)
    p_value, real_mean = pipeline_stats.calculate_combine_pvalues(by_recall_de_dict)
    print (p_value,real_mean)
    p_value, real_mean = pipeline_stats.calculate_combine_pvalues(by_fscore_de_dict)
    print(p_value,real_mean)

    first_sents = []
    for de_index, alignment in by_first_de_dict.items():
        # assert (alignment._en_index == de_index*100)
        first_sents.append((alignment._en_sent, str(alignment._precision), str(alignment._recall), str(alignment._f_score)))

    with open("pipeline_results_long_take_the_first", "w") as f:
        f.write("\n".join([tup[0] for tup in first_sents]))
    pipeline_stats.calculate_bleu_from_mteval("pipeline_results_long_gt_filtered", "pipeline_results_long_take_the_first")
    with open("pipeline_results_long_take_the_first", "w") as f:
        f.write("\n".join(["\t".join(tup) for tup in first_sents]))

    gt_sents = [sent for i, sent in enumerate(gt_sents) if i not in [a._de_index for a in without_predicates]]

    name = ["precision", "recall", "f_score"]
    for i,dict in enumerate([by_precision_de_dict, by_recall_de_dict, by_fscore_de_dict]):
        sents = []
        scores = []
        for item in dict:
            max_score, max_list = max(dict[item].items(), key=operator.itemgetter(0))
            # print(max_list[0]._en_index)
            # print(max_list[0]._score, " ", len(max_list))
            sents.append(max_list[0]._en_sent)
            scores.append(max_score)

        with open("pipeline_results_long_"+name[i], "w") as f:
            f.write("\n".join(sents))

        print(name[i])
        pipeline_stats.calculate_bleu_from_mteval("pipeline_results_long_gt_filtered", "pipeline_results_long_"+name[i])
        with open("pipeline_results_long_"+name[i], "w") as f:
            f.write("\n".join([a +"\t"+ str(b) for a,b in zip(sents,scores)]))


        unique_indices = pipeline_stats.calculate_stats(gt_sents,[tup[0] for tup in first_sents],sents)
        all_unique_indices += unique_indices
        # unique_sents = [(i,sents[i]) for i in range(len(sents)) if i in unique_indices]
        # unique_scores = [(i,scores[i]) for i in range(len(scores)) if i in unique_indices]
        # with open("pipeline_results_long_different_" + name[i], "w") as f:
        #     f.write("\n".join([str(a) + "\t" + str(b[1:]) for a, b in zip(unique_sents, unique_scores)]))

    return set(all_unique_indices)

if __name__ == "__main__":
    # en = spacy.load("en")
    # de = spacy.load("de")
    # sent = en("she and her mother were absolutely best friends .")
    # fix_en_args(sent)
    # print(create_tree_for_latex({token.i: (token.head.i, token.dep_) for token in sent}, sent.string))
    # de_sent = de("sie und ihre Mutter waren absolut beste Freunde .")
    # fix_de_args(de_sent)
    # print(create_tree_for_latex({token.i : (token.head.i, token.dep_)for token in de_sent}, de_sent.string))

    # build_project()
    all_unique_indices = run_pipeline()
    organize_results(all_unique_indices)