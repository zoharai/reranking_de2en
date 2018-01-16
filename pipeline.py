# End to end pipeline for reranking traslated de-en sentences.
# author - Zohar Aizenbud

import spacy
from comparison_v2 import Alignment
import bsbleu
import pickle
import operator
from forAligner import add_spaces_to_hypen

# from spacy.tokens import Doc
# from spacy.vocab import Vocab

def calculate_bleu_score(R,C, files=False):
    return bsbleu.bleu(R,C, False, files=files, bootstrap=50)


def fix_ge_args(de_doc):
    de_args = {}
    for token in de_doc:
        if token.pos_ == "VERB" and token.head != token and (token.head.tag_.startswith("VA") ): #or token.head.tag_.startswith("VM")):
            de_args[token.head.i] = (token.i, token.dep_)
            if token.head.dep_ == "ROOT":
                de_args[token.i] = (token.i, "ROOT")
            else:
                de_args[token.i] = (token.head.head.i, token.head.dep_)

            for child in token.head.children:
                if child != token:
                    de_args[child.i] = (token.i, child.dep_)
            # print ("here")

    for token in de_doc:
        if token.i not in de_args:
            de_args[token.i] = (token.head.i, token.dep_)

    return de_args

def build():
    alignments = []
    alignments_dict = {}
    en_model = spacy.load("en")
    de_model = spacy.load("de")

    #build alignments objects
    with open("test_alignments_new", "r") as f:
        prev_de_sent = ""
        de_args = {}
        j = -1
        lines = f.readlines()
        for i, line in enumerate(lines):
            #print(i)
            de, en, align, score = line.split("|||")
            if de !=prev_de_sent:
                de_doc = de_model(de.strip())
                de_args = fix_ge_args(de_doc)

                j+=1
                prev_de_sent = de

            en_doc = en_model(en.strip())
            en_args = {token.i: (token.head.i, token.dep_) for token in en_doc}
            a = Alignment(en.strip(),de.strip(),align, i, j,en_args ,de_args)
            # a.comparison()
            alignments.append(a)
            alignments_dict[(j,i)] = a

    # add gt english sentences
    with open("newstest2016.tc.en") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            for j in range(int(i) * 100, (int(i) + 1) * 100):
                if (int(i), j) in alignments_dict:
                    alignments_dict[(int(i), j)].update_gt_en_sent(add_spaces_to_hypen(line.strip().replace("``", '"').replace("''" ,'"')))


    # save to pickle:
    with open('alignments_data.pkl', 'wb') as output:
        for a in alignments:
            pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

    return alignments, alignments_dict


def stanford_build():
    alignments = []
    alignments_dict = {}

    # build alignments objects
    with open("test_alignments", "r") as f:
        german_lines = process_output_dependency_conll("newstest2016.tc.de.stanford.conll")
        english_lines = process_output_dependency_conll("newstest2016.en.output.sentences.nbest.stanford.conll")
        prev_de_sent = ""
        j = -1
        lines = f.readlines()
        for i, line in enumerate(lines):
            # print(i)
            de, en, align, score = line.split("|||")
            if de != prev_de_sent:
                j += 1
                prev_de_sent = de

            en_args = english_lines[i]
            de_args = german_lines[j]
            a = Alignment(en.strip(), de.strip(), align, i, j, en_args, de_args, conll=True)
            alignments.append(a)
            alignments_dict[(j, i)] = a

        # add gt english sentences
    with open("newstest2016.tc.en") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            for j in range(int(i) * 100, (int(i) + 1) * 100):
                if (int(i), j) in alignments_dict:
                    alignments_dict[(int(i), j)].update_gt_en_sent(line.strip())

    with open('stanford_alignments_data.pkl', 'wb') as output:
        for a in alignments:
            pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

    return alignments, alignments_dict


def process_output_dependency_conll(file):
    args_lines = []
    with open(file, "r") as input:
        lines = input.readlines()
        i = 0
        sent = []
        args = {}
        sent_index = 0
        while i < len(lines):
            if lines[i] == "\n":
                args_lines.append(args)
                sent_index += 1
                sent = []
                args = {}
            else:
                words = lines[i].split()
                assert (len(words) == 7)
                index = words[0]
                word = words[1]
                head = words[5]
                func = words[6]
                sent.append(word)
                args[int(index)-1] =  (int(head)-1, func)
            i+=1
    return args_lines


def run_pipeline():
    # #len(alignments) = 24281

    # calculate_bleu_score(["newstest2016.tc.en.short"], ["newstest2016.en.output.nbest_first_one"],files=True)
    by_precision_de_dict = {}
    by_recall_de_dict = {}
    by_fscore_de_dict = {}
    by_first_de_dict = {}
    with open('alignments_data.pkl', 'rb') as input:
       for i in range(24281):
            # print(i)
            a = pickle.load(input)
            a.comparison()
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

    gt_sents = []
    first_sents = []
    for de_index, alignment in by_first_de_dict.items():
        assert (alignment._en_index == de_index*100)
        gt_sents.append(alignment._gt_en_sent)
        first_sents.append(alignment._en_sent)

    calculate_bleu_score(gt_sents, first_sents)
    with open("pipeline_results_take_the_first", "w") as f:
        f.write("\n".join(first_sents))

    name = ["precision", "recall", "f_score"]
    for i,dict in enumerate([by_precision_de_dict, by_recall_de_dict, by_fscore_de_dict]):
        gt_sents = []
        sents = []
        for item in dict:
            max_list = max(dict[item].items(), key=operator.itemgetter(0))[1]
            # print(max_list[0]._en_index)
            # print(max_list[0]._score, " ", len(max_list))
            gt_sents.append(max_list[0]._gt_en_sent)
            sents.append(max_list[0]._en_sent)
            # for x in max_list:
                # print(x._en_sent)



        indexes = [i for i in range(len(sents)) if sents[i]!= first_sents[i]]
        # calculate_bleu_score([gt_sents[i] for i in range(len(gt_sents)) if i  in indexes ], [first_sents[i] for i in range(len(first_sents)) if i in indexes ])
        # calculate_bleu_score([gt_sents[i] for i in range(len(gt_sents)) if i  in indexes ], [sents[i] for i in range(len(sents)) if i  in indexes ])
        calculate_bleu_score(gt_sents, sents)
        with open("pipeline_results_"+name[i], "w") as f:
            f.write("\n".join(sents))


if __name__ == "__main__":

    # build()
    run_pipeline()