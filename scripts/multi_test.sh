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
