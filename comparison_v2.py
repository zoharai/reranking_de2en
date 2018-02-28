
en_predicate_arguments = [
'csubj',
'csubjpass',
# 'agent',
'expl',
'nsubj',
'nsubjpass',
'xcomp',
'ccomp',
'attr',
'dobj',
'iobj',
'oprd',
'obj',
'dative',

#new
'pverb',
'npadvmod',
# 'acomp',
]

de_predicate_arguments = [
'oa',
'oc',
'og',
'op',
'da',
'pd',
'sb',
'rs',
#new
'verbmo',
'pverb'
]



class Alignment():
    def __init__(self, en, de, align, i, j, en_args, de_args, en_prob, conll=False):
        self._en_index = i
        self._en_prob = en_prob
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
        self._without_predicate = False


    def update_gt_en_sent(self, sent, args):
        self._gt_en_sent = sent
        self._gt_en_words = sent.split()
        self._gt_en_args = args


    def get_german_sent(self):
        return self._de_sent

    def get_english_sent(self):
        return self._en_sent


    def comparison(self):
        de_predicates = [(arg,value[0]) for arg, value in self._de_args.items() if len(set(value[1].split("||")).intersection(de_predicate_arguments)) != 0]
        if len(de_predicates) == 0:
            self._without_predicate = True
            return

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

        #no case of 1 to many for en.
        if len(en_predicates_aligned) != 0:
            self._precision = len(en_good_predicates)/len(en_predicates_aligned)

        if len(de_predicates) != 0:
            self._recall = len(en_good_predicates)/len(de_predicates)

        if self._precision !=0 or self._recall != 0:
            self._f_score = (2*self._precision*self._recall)/(self._precision + self._recall)







