from collections import defaultdict
def add_spaces_to_hypen(sent, with_hypen):
    words = sent.split()
    new_words = []
    for word in words:
        hypen_words = word.split("-")
        if len(hypen_words) == 1:
            new_words.append(word)
        else:
            for i,h_word in enumerate(hypen_words):
                new_words.append(h_word)
                if i != len(hypen_words)-1:
                    if with_hypen:
                        new_words.append("-")

    #fix O ' WORD problem
    i=0
    fixed_words = []
    while i < len(new_words):
        if i < len(words)-2 and new_words[i] == "O" and (new_words[i + 1] == "'" or new_words[i + 1] == "&apos;"):
            fixed_words.append(new_words[i] + "'" + new_words[i + 2])
            i+=3
        elif i < len(words)-1  and new_words[i] == "O" and new_words[i+1].startswith("'"):
            fixed_words.append(new_words[i]+new_words[i+1])
            i+=2
        else:
            fixed_words.append(new_words[i])
            i+=1


    new_sent = " ".join(fixed_words)




    return new_sent


def output_sentences(with_bpe):
    with open("/home/zohar/Documents/Master/NLP_Lab/pipeline_data/newstest2016.parzu.no-a.factors.1.de.output.ensemble_best.nbest.test", "r") as input:
        with open("newstest2016.en.output.sentences_indexes.long.nbest", "w") as output:
            with open("newstest2016.en.output.sentences.long.nbest", "w") as pos_output:
                with open("newstest2016.en.output.probability.long.nbest", "w") as prob_output:
                    for line in input.readlines():
                        index, sentence, prob = line.split("|||")
                        words = [x for x in sentence.split() if not x.startswith("<")]
                        new_words = []
                        if (with_bpe):
                            new_words = words
                        else:
                            for i in range(len(words)):
                                if not words[i].endswith("@"):
                                    new_words.append(words[i])
                                else:
                                    word = words[i].replace("@", "")
                                    if i+1 < len(words):
                                        words[i+1] = word + words[i+1]
                                    else:
                                        new_words.append(word)

                        output.write(index +"|||" + " ".join(new_words)+ "\n")
                        pos_output.write(" ".join(new_words) + "\n")
                        prob_output.write(prob)




def align_test(with_hypen):
    with open("newstest2016.de.tc", "r") as input:
        with open("newstest2016.en.output.sentences_indexes.long.nbest", "r") as english_input:
            with open("newstest2016.de-en.nbest.long.NO-DUPLICATES.aligned", "w") as output:

                german = input.readlines()
                german_dict = defaultdict(list)
                for i in range(len(german)):
                    german[i] = add_spaces_to_hypen(german[i].replace("``", '"').replace("''" ,'"').replace(",", ""), with_hypen)

                for ind, line in enumerate(english_input.readlines()):

                    (num, sent) = line.split("|||")
                    sent = add_spaces_to_hypen(sent.replace("``", '"').replace("''" ,'"').replace(",", ""), with_hypen)
                    german_ind = int(num)
                    if not sent in german_dict[german_ind]:
                        german_dict[german_ind].append(sent)
                        output.write(german[german_ind] +" ||| "+ sent+"\n")


def align_gt(with_hypen):
    with open("newstest2016.de.tc", "r") as input:
        with open("newstest2016.tc.en", "r") as english_input:
            with open("newstest2016.de-en.gt.long.NO-HYPEN.aligned", "w") as output:

                german = input.readlines()
                for i in range(len(german)):
                    german[i] = add_spaces_to_hypen(german[i].replace("``", '"').replace("''", '"').replace(",", ""), with_hypen)

                for ind, line in enumerate(english_input.readlines()):
                    sent = add_spaces_to_hypen(line.replace("``", '"').replace("''", '"').replace(",", ""), with_hypen)
                    output.write(german[ind] + " ||| " + sent + "\n")

# wmt16.parallel.en-de.tc.no-n.aligned - a version without error lines - lines with just english or german
# there is script for fixing the problem in data/train/check_aligned.py
def align_train(with_hypen):
    with open("/home/zohar/Documents/Master/NLP_Lab/pipeline_data/wmt16.parallel.en-de.tc.no-n.aligned", "r") as input:
            with open("/home/zohar/Documents/Master/NLP_Lab/pipeline_data/wmt16.parallel.en-de.tc.no-n.NO-HYPEN.aligned", "w") as output:


                for i,line in enumerate(input.readlines()):

                    line = add_spaces_to_hypen(line.replace("``", '"').replace("''" ,'"').replace(",", ""), with_hypen)
                    if line.strip() != "|||":
                        output.write(line+"\n")


if __name__ == "__main__":
    align_gt(False)
    # align_train(False)
    # output_sentences(False)
    align_test(False)