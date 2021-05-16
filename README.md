# Multi-Task Parser
This repo is a fork of the SuPar repo https://github.com/yzhangcs/parser. For instructions on how to use parsers found in the base repo, please see their README. We focus here on modifications we made and the relevant instructions for using the multi-task parser.


## Training
To run the multi-task parser using two treebanks with default parameters and settings use the following command:

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile \    --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name

where --train, --dev, and --test requiring the train, dev, and test files respectively each desired treebank. Each treebank then needs a designated task name specified by the --task-names argument.

Note that the multi-task parser can input as many treebanks as one so chooses by simply listing additional treebanks and tasks (and example with three is provided below).

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile path/to/treebank3/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile path/to/treebank3/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile path/to/treebank3/testfile --task-names task1name task2name task3name


#### Training with only randomly initialized word embeddings

The parser has been modified to allow for the usage of only randomly initialized word embeddings. This can be done by the using -f None as seen below:

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name -f None

#### Training with shared MLP layers

The default behavior of the parser is to train without sharing the MLP layers between treebanks. To train sharing the MLP layers use the --share-mlp argument.

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name --share-mlp

#### Training with a joint loss

The default behavior of the parser is to train using alternating batch loss. There is the possibility of training with the parser with a join loss using --joint-loss.

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name --joint-loss

Note: to use a joint loss, the treebanks must be parallel treebanks (e.g. UD and SUD version of a treebank).

#### Training with loss weights

To assign specific loss weights to an individual task, use the --loss-weights argument and then specicfy the weight for each task in the order that the tasks are provided:

    python -m supar.cmds.multi_parser train -b -d 0 -c config.ini -p path/to/experiment/directory --train path/to/treebank1/trainfile path/to/treebank2/trainfile --dev path/to/treebank1/devfile path/to/treebank2/devfile --test path/to/treebank1/testfile path/to/treebank2/testfile --task-names task1name task2name --loss-weights 0.95 0.05

Note: the default behavior is to assign 1.0 to all tasks and is currently availble only for alternating batch losss.

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
