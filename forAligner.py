def add_spaces_to_hypen(sent):
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
                    new_words.append("-")

    new_sent = " ".join(new_words)
    return new_sent


def align_test():
    with open("newstest2016.de.tc", "r") as input:
        with open("newstest2016.en.output.sentences_indexes.nbest", "r") as english_input:
            with open("newstest2016.de-en.nbest.aligned", "w") as output:

                german = input.readlines()
                for i in range(len(german)):
                    german[i] = add_spaces_to_hypen(german[i].replace("``", '"').replace("''" ,'"'))

                for ind, line in enumerate(english_input.readlines()):

                    (num, sent) = line.split("|||")
                    sent = add_spaces_to_hypen(sent.replace("``", '"').replace("''" ,'"'))
                    output.write(german[int(num)][:-1] +" ||| "+ sent+"\n")


#wmt16.parallel.en-de.tc.no-n.aligned - a version withour error lines - lines with just english or german
# there is script for fixing the problem in data/train/check_aligned.py
def align_train():
    with open("wmt16.parallel.en-de.tc.no-n.aligned", "r") as input:
            with open("wmt16.parallel.en-de.tc.no-n.fixed.aligned", "w") as output:


                for line in input.readlines():

                    line = add_spaces_to_hypen(line.replace("``", '"').replace("''" ,'"'))
                    output.write(line+"\n")


# align_train()