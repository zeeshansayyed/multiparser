python -m supar.cmds.multi_parser \
    train -b -d 1 -c config2.ini \
    -p exp/arabic/ud-sud/test2 \
    -f tag \
    --task-names ud sud \
    --train \
    data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu \
    data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-train.conllu \
    --dev \
    data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu \
    data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu \
    --test \
    data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu \
    data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu \
    --n-embed 300 \
    --batch-size 20000 \
    --patience 50
# --embed data/embeddings/cc.ar.300.vec \


# Example Train Scripts
python -m supar.cmds.multi_parser     train -b -d 1 -c config2.ini     -p exp/arabic/ud-sud/shared-biaffine-lstm4     -f tag     --task-names ud sud     --train     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-train.conllu     --dev     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu     --test     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu     --n-embed 300     --batch-size 20000     --patience 100 --joint-loss
python -m supar.cmds.multi_parser     train -b -d 0 -c config.ini     -p exp/arabic/ud-sud/simple-share     -f tag     --task-names ud sud     --train     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-train.conllu     --dev     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu     --test     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu     --n-embed 300     --batch-size 20000     --patience 100 --share-mlp
python -m supar.cmds.multi_parser     train -b -d 0 -c config.ini     -p exp/arabic/ud-sud/test-3     -f tag     --task-names ud sud     --train     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-train.conllu     --dev     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu     --test     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu     data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu     --n-embed 300     --batch-size 20000     --patience 100  --optimizer-type multiple --finetune whole --train-mode finetune --optimizer-type multiple
#Example Evaluate scripts
python -m supar.cmds.multi_parser evaluate -d 0 -c config.ini -p exp/arabic/ud-sud/simple-share-1/ud.model --task-names ud --data data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu --pred exp/arabic/ud-sud/simple-share-1/
python -m supar.cmds.multi_parser evaluate --proj -d 0 -c config.ini -p exp/arabic/ud-sud/alternating-partial-multiple-nosharemlp/total.model --task ud --data data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu --task sud --data data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu

# Example Predict scrupts
python -m supar.cmds.multi_parser predict --tree --proj -d 0 -c config.ini -p exp/arabic/ud-sud/simple-share-1/ud.model --task-names ud --data data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu --pred exp/arabic/ud-sud/simple-share-1/ud-test.conllx

# Finetuning
python -m supar.cmds.multi_parser     finetune  -d 0  -p exp/arabic/ud-sud/simple-share-2/total.model         --task-names ud     --train     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu         --dev     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu     --test     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu      --batch-size 20000     --patience 10 -lr 2e-4
python -m supar.cmds.multi_parser     finetune  -d 0  -p exp/arabic/ud-sud/simple-share-2/ud.model         --task-names ud     --train     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu         --dev     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu     --test     data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu      --batch-size 20000     --patience 10 -lr 3e-5
