import pprint
import pickle
import re
import operator
import nltk
import spacy

nlp = spacy.load('en')
doc = nlp(u'Autonomous cars shift insurance liability toward manufacturers')
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)


def bleu(gt, sent):
    return nltk.translate.bleu_score.sentence_bleu([gt], sent)


predicate_arguments = [
    'subj',
'obja',
#'aux',
'pred',
'objc',
'objp',
'objd',
's',
'obji',
'subjc',
'objg']

class Alignment():
    def __init__(self, en, de, align, i, j, _en_args, _de_args):
        self._en_index = i
        self._de_index = j
        self._en_sent = en
        self._de_sent = de.replace("``", '"').replace("''" ,'"')
        self._en_words = en.split()
        self._de_words  = de.split()
        self._de_en_alignments = [(int(part.split("-")[0]), int(part.split("-")[1])) for part in align.split()]
        self._de_args = {}
        self._en_args = {}
        self._de_args_count = 0
        self._en_args_count = 0
        #if score == -1 - there is a failure with building the object.
        self._score = 0
        self._gt_en_sent = ""
        self._gt_en_words = []

    def update_gt_en_sent(self, sent):
        self._gt_en_sent = sent
        self._gt_en_words = sent.split()

    def update_en_args(self, index, head, function):
        #assert(index not in self._en_args)
        if index not in self._en_args:
            self._en_args[index] = []
        self._en_args[index].append((head, function))

    def update_de_args(self, index, head, function):
        #assert (index not in self._de_args)
        if index not in self._de_args:
            self._de_args[index] = []
        self._de_args[index].append((head, function))

    def caclulate_score(self):
        self._score = 0

    def update_score(self, new_score):
        self._score = new_score

    def get_german_sent(self):
        return self._de_sent

    def get_english_sent(self):
        return self._en_sent


    def comparison(self):
        if self._score == -1:
            return

        for arg,values in self._de_args.items():
            for value in values:
                if value[1] in predicate_arguments:
                    self._de_args_count += 1
                    de_index = arg
                    de_head_index = value[0]
                    en_indexes = [tup[1] for tup in self._de_en_alignments if tup[0] == de_index]
                    en_head_indexes = [tup[1] for tup in self._de_en_alignments if tup[0] == de_head_index]
                    en_head_poss = [tup[0] for en_index in en_indexes if en_index in self._en_args for tup in self._en_args[en_index] if tup[0] in en_head_indexes]
                    if len(en_head_poss) != 0:
                        #print("len ", len(en_head_poss))
                        self._score += 1

        pattern = re.compile(r"(.*S.*)(/|\\)\1")
        for arg, values in self._en_args.items():
            for value in values:
                supertag = value[-1]

                if "S" in supertag and pattern.search(supertag) == None:
                    self._en_args_count +=1
                    #print(self._en_index, " ", self._en_sent, " ", self._en_words[arg], " ", self._en_words[value[0]], " ", supertag)

        # #ONE OPTION:
        # if self._en_args_count != self._de_args_count:
        #     self._score -= abs(self._en_args_count-self._de_args_count)

                    # # ANOTHER OPTIONS:
                    # en_index = arg
                    # en_head_index = value[0]
                    # de_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_index]
                    # de_head_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_head_index]
                    # de_head_poss = [tup[0] for de_index in de_indexes if de_index in self._de_args for tup in
                    #                 self._de_args[de_index] if tup[0] in de_head_indexes]
                    # if len(de_head_poss) == 0:
                    #     # print("len ", len(en_head_poss))
                    #     self._score += -1





        # if self._score > 4:# and self._en_sent.startswith("46"):
        #     print(self._de_sent)
        #     print(self._en_sent)
        #     print ("score ", self._score)








def build():
    alignments = []
    alignments_dict = {}
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
            a = Alignment(en.strip(),de.strip(),align, i, j)
            alignments.append(a)
            alignments_dict[(j,i)] = a

    # add gt english sentences
    with open("/home/zohar/Documents/Master/NLP_Lab/LEXI/newstest2016.tc.en") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            for j in range(int(i) * 100, (int(i) + 1) * 100):
                if (int(i), j) in alignments_dict:
                    alignments_dict[(int(i), j)].update_gt_en_sent(line.strip())


    #update english deps
    with open("newstest2016.en.nbest.dep.processed", "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].find("|||")  != -1:
                en_index, en_sent = lines[i].split(" ||| ")
                print(en_index)
                if en_sent == "fail\n":
                    alignments[int(en_index) - 1].update_score(-1)
                    i+=2
                else:
                    i+=1
                    while lines[i] != "******************\n":
                        index, word, head, function, supertag = lines[i].split()
                        assert  alignments[int(en_index)-1].get_english_sent() == en_sent.strip()
                        alignments[int(en_index)-1].update_en_args(int(index), int(head), function, supertag)
                        i += 1
                    i+=1
    #update german deps
    with open("newstest2016.de.tc.conll.dep.processed", "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].find("|||")  != -1:
                de_index, de_sent = lines[i].split(" ||| ")

                i+=1
                while lines[i] != "******************\n":
                    index, word, head, function = lines[i].split()
                    for j in range(int(de_index)*100, (int(de_index)+1)*100):
                        if (int(de_index),j) in alignments_dict:
                            assert alignments_dict[(int(de_index),j)].get_german_sent() == de_sent.strip()
                            print(de_index)
                            alignments_dict[(int(de_index),j)].update_de_args(int(index), int(head), function)
                    i += 1
                i+=1






    #save to pickle:
    with open('alignments_data.pkl', 'wb') as output:
        for a in alignments:
            pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

    print(len(alignments))


# # build()
# #len(alignments) = 24281
# #read from pickle:
# by_de_dict = {}
# alignments = []
# with open('alignments_data.pkl', 'rb') as input:
#    for i in range(24281):
#         # print(i)
#         a = pickle.load(input)
#         # print(a._de_sent)
#         # print(a._en_sent)
#         a.comparison()
#         #alignments.append(a)
#         if a._de_index not in by_de_dict:
#             by_de_dict[a._de_index] = {}
#         if a._score not in by_de_dict[a._de_index]:
#             by_de_dict[a._de_index][a._score] = []
#         by_de_dict[a._de_index][a._score].append(a)
#
#
# gt_sents = []
# sents = []
# for item in by_de_dict:
#     max_list = max(by_de_dict[item].items(), key=operator.itemgetter(0))[1]
#     print(max_list[0]._de_sent)
#     print(max_list[0]._score)
#     gt_sents.append(max_list[0]._gt_en_sent)
#     sents.append(max_list[0]._en_sent)
#     # for x in max_list:
#         # print(x._en_sent)
#         # print(bleu(x._gt_en_sent, x._en_sent))
#
# print(nltk.translate.bleu_score.corpus_bleu(gt_sents, sents))
# print ("\n".join(sents))
# # print(len(alignments))



