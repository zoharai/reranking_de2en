# End to end pipeline for reranking traslated de-en sentences.
# author - Zohar Aizenbud

import spacy
from comparison_v2 import Alignment
import bsbleu
import pickle
import operator
# from spacy.tokens import Doc
# from spacy.vocab import Vocab

def calculate_bleu_score(R,C, files=False):
    return bsbleu.bleu(R,C, False, files=files)



# def build_dep_tree(en_model, de_model, alignment):
#     en_sent_doc = en_model(alignment._en_sent)
#     for index, token in enumerate(en_sent_doc):
#         print (token)
#         a.update_en_args(i, token.dep_ , token.dep_)
#     de_sent_doc = de_model(alignment._de_sent)


def build():
    alignments = []
    alignments_dict = {}
    en_model = spacy.load("en")
    de_model = spacy.load("de")

    #build alignments objects
    with open("test_alignments", "r") as f:
        prev_de_sent = ""
        j=-1
        lines = f.readlines()
        for i, line in enumerate(lines):
            #print(i)
            de, en, align, score = line.split("|||")
            if de !=prev_de_sent:
                j+=1
                prev_de_sent = de

            en_doc = en_model(en.strip())
            de_doc = de_model(de.strip())
            # en_bytes = en_doc.to_bytes()
            # de_bytes = de_doc.to_bytes()
            a = Alignment(en.strip(),de.strip(),align, i, j,en_doc ,de_doc)
            alignments.append(a)
            alignments_dict[(j,i)] = a

    # add gt english sentences
    with open("newstest2016.tc.en") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            for j in range(int(i) * 100, (int(i) + 1) * 100):
                if (int(i), j) in alignments_dict:
                    alignments_dict[(int(i), j)].update_gt_en_sent(line.strip())


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


    # with open("newstest2016.en.output.sentences_indexes.nbest") as file:
    #     with open("newstest2016.en.output.nbest_first_one", "w") as out:
    #         prev_num = ""
    #         for line in file.readlines():
    #             num, sent = line.split(" |||")
    #             if num != prev_num:
    #                 out.write(sent)
    #                 prev_num = num

    # alignments, alignments_dict = build()
    # for alignment in alignments:
    #     alignment.comparison()

    # #len(alignments) = 24281

    calculate_bleu_score(["newstest2016.tc.en.short"], ["newstest2016.en.output.nbest_first_one"],files=True)


    by_de_dict = {}
    with open('alignments_data.pkl', 'rb') as input:
       for i in range(24281):
            # print(i)
            a = pickle.load(input)
            a.comparison()
            if a._de_index not in by_de_dict:
                   by_de_dict[a._de_index] = {}
            if a._score not in by_de_dict[a._de_index]:
                by_de_dict[a._de_index][a._score] = []
            by_de_dict[a._de_index][a._score].append(a)

    gt_sents = []
    sents = []
    for item in by_de_dict:
        max_list = max(by_de_dict[item].items(), key=operator.itemgetter(0))[1]
        # print(max_list[0]._en_index)
        # print(max_list[0]._score, " ", len(max_list))
        gt_sents.append(max_list[0]._gt_en_sent)
        sents.append(max_list[0]._en_sent)
        # for x in max_list:
            # print(x._en_sent)

    calculate_bleu_score(gt_sents, sents)
    # with open("pipeline_results_negative_score_stanford", "w") as f:
    #     f.write("\n".join(sents))


if __name__ == "__main__":
    run_pipeline()