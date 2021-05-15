# Multi-Task Parser
This Repo is a fork of the SuPar repo https://github.com/yzhangcs/parser. For instructions on how to use parsers found in the base repo, please see their README. We focus here on modifications we made and the relevant instructions.


## Training
To run the multi-task parser using two treebanks with default parameters and settings use the following command:

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile \    --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name

where --train, --dev, and --test requiring the train, dev, and test files respectively each desired treebank. Each treebank then needs a designated task name specified by the --task-names argument.

Note that the multi-task parser can input as many treebanks as one so chooses by simply listing additional treebanks and tasks (and example with three is provided below).

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile path/to/treebank3/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile path/to/treebank3/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile path/to/treebank3/testfile --task-names task1name task2name task3name


#### Training with only randomly initialized word embeddings

The parser has been modified to allow for the usage of only randomly initialized word embeddings. This can be done by the using -f None as seen below:

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name -f None

## Cite

If you use the multi-task parser, pease cite:

```
@InProceedings{alisayyed:dakota:2021,
  Title                    = {Annotations Matter: Leveraging Multitask Learning to Parse UD and SUD},
  Author                   = {Ali Sayyed, Zeeshan and Dakota, Daniel},
  Booktitle                = {Findings of the Association for Computational Linguistics: ACL 2021},
  Year                     = {2021}
}
```
