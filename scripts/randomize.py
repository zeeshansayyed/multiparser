import sys
import random

num = 909
with open(sys.argv[1], "r") as inputFile:

    sentences = inputFile.read().split("\n\n")
    print (len(sentences))
    #print (sentences[0])
    random.shuffle(sentences)

    with open(sys.argv[1]+"_"+str(num), "w") as outputFile:
        for sentence in sentences[:num]:
            for s in sentence:
                outputFile.write(s)
            outputFile.write("\n\n")
        outputFile.close()

