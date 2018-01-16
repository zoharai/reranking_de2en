
en_predicate_arguments = [
    'csubj',
'csubjpass',
'agent',
'expl',
'nsubj',
'nsubjpass',

'attr',
'dobj',
'iobj',
'oprd',

'obj',


# 'pobj'
]

de_predicate_arguments = [
'oa',
'oc',
'og',
'op',

'pd',
'sb']

stanford_predicate_arguments = [
    'agent',
    'csubj',
    'csubjpass',
    'dobj',
    'expl', #expletive?
    'iobj',
    'nsubj',
    'nsubjpass',
    'xcomp', #open clausal complement?
    'xsubj' , #controlling subject?

]

class Alignment():
    def __init__(self, en, de, align, i, j, en_args, de_args, conll=False):
        self._en_index = i
        self._de_index = j
        self._en_sent = en
        self._de_sent = de.replace("``", '"').replace("''" ,'"')
        self._en_words = en.split()
        self._de_words  = de.split()
        self._de_en_alignments = [(int(part.split("-")[0]), int(part.split("-")[1])) for part in align.split()]
        self._de_args = de_args
        self._en_args = en_args

        self._de_args_count = 0
        self._en_args_count = 0
        #if score == -1 - there is a failure with building the object.
        self._score = 0
        self._gt_en_sent = ""
        self._gt_en_words = []
        self._precision = 0
        self._recall = 0
        self._f_score = 0


    def update_gt_en_sent(self, sent):
        self._gt_en_sent = sent
        self._gt_en_words = sent.split()


    def get_german_sent(self):
        return self._de_sent

    def get_english_sent(self):
        return self._en_sent


    def comparison(self):
        de_predicates = [(arg,value[0]) for arg, value in self._de_args.items() if len(set(value[1].split("||")).intersection(de_predicate_arguments)) != 0]
        en_predicates = {arg: value for arg, value in self._en_args.items() if value[1] in en_predicate_arguments}

        en_predicates_aligned = []
        for arg, value in en_predicates.items():
            en_index = arg
            en_head_index = value[0]
            de_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_index]
            de_head_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_head_index]
            for de_index in de_indexes:
                for de_head_index in de_head_indexes:
                    en_predicates_aligned.append((de_index, de_head_index))

        en_predicates_aligned = list(set(en_predicates_aligned))
        en_good_predicates = list(set([tup for tup in en_predicates_aligned if tup in de_predicates]))

        if len(en_predicates_aligned) != 0:
            self._precision = len(en_good_predicates)/len(en_predicates_aligned)

        if len(de_predicates) != 0:
            self._recall = len(en_good_predicates)/len(de_predicates)

        if self._precision !=0 or self._recall != 0:
            self._f_score = (2*self._precision*self._recall)/(self._precision + self._recall)

        # for arg, value in self._de_args.items():
        #     if value[1] in de_predicate_arguments:
        #         self._de_args_count += 1
        #         de_index = arg
        #         de_head_index = value[0]
        #         en_indexes = [tup[1] for tup in self._de_en_alignments if tup[0] == de_index]
        #         en_head_indexes = [tup[1] for tup in self._de_en_alignments if tup[0] == de_head_index]
        #         en_head_poss = [ self._en_args[en_index][0]
        #                          for en_index in en_indexes
        #                          if en_index in self._en_args and
        #                          self._en_args[en_index][0] in en_head_indexes
        #                          and self._en_args[en_index][1] in en_predicate_arguments]
        #         if len(en_head_poss) != 0:
        #             # print("len ", len(en_head_poss))
        #             self._score += 1
        #         # if len(en_head_poss) == 0:
        #         #     # print("len ", len(en_head_poss))
        #         #     self._score += -1
        #
        #
        #
        # for arg, value in self._en_args.items():
        #     if value[1] in en_predicate_arguments:
        #         if value[1] == "obj":
        #             print("here")
        #         en_index = arg
        #         en_head_index = value[0]
        #         de_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_index]
        #         de_head_indexes = [tup[0] for tup in self._de_en_alignments if tup[1] == en_head_index]
        #         de_head_poss = [ self._de_args[de_index][0]
        #                          for de_index in de_indexes
        #                          if de_index in self._de_args and
        #                          self._de_args[de_index][0] in de_head_indexes
        #                          and self._de_args[de_index][1] in de_predicate_arguments]
        #         if len(de_head_poss) == 0:
        #             # print("len ", len(en_head_poss))
        #             self._score += -1
        #
        #         # # second option:
        #         # if len(de_head_poss) != 0:
        #         #     # print("len ", len(en_head_poss))
        #         #     self._score += 1




