CUDA_VISIBLE_DEVICES=0
# python -m supar.cmds.biaffine_dependency train -b -d 0 -c config.ini -p exp/ptb/model -f char --train data/ptb/ptb_train_3.3.0.sd.clean --dev data/ptb/ptb_dev_3.3.0.sd.clean --test data/ptb/ptb_test_3.3.0.sd.clean

# SPMRL splits for Arabic
# python -m supar.cmds.biaffine_dependency train -b -d 0 -c config.ini \
#     -p exp/atb/depmodel -f char --train data/atb/train.arabic.gold.conll \
#     --dev data/atb/dev.arabic.gold.conll --test data/atb/test.arabic.gold.conll \
#     --embed data/embeddings/cc.ar.300.vec --n-embed 300 \
#     --batch-size 40000
python -m supar.cmds.biaffine_dependency predict \
    -d 0 \
    -c config.ini \
    -p exp/arabic/spmrl/biaff-dep \
    --data data/atb/test.arabic.gold.conll \
    --pred exp/arabic/spmrl/biaff-dep.spmrl.pred.conllx

# Arabic UD - Training
# python -m supar.cmds.biaffine_dependency \
#     train -b -d 0 -c config.ini \
#     -p exp/arabic/ud/biaff-dep \
#     -f char \
#     --train data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu \
#     --dev data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu \
#     --test data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu \
#     --embed data/embeddings/cc.ar.300.vec \
#     --n-embed 300 \
#     --batch-size 40000
# Arabic UD - Predict
python -m supar.cmds.biaffine_dependency predict \
    -d 0 \
    -c config.ini \
    -p exp/arabic/ud/biaff-dep \
    --data data/ud/ud-treebanks-v2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu \
    --pred exp/arabic/ud/biaff-dep.ud.pred.conllx

# Arabic SUD
# python -m supar.cmds.biaffine_dependency \
#     train -b -d 0 -c config.ini \
#     -p exp/arabic/sud/biaff-dep \
#     -f char \
#     --train data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-train.conllu \
#     --dev data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-dev.conllu \
#     --test data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu \
#     --embed data/embeddings/cc.ar.300.vec \
#     --n-embed 300 \
#     --batch-size 40000
# Arabic SUD - Predict
python -m supar.cmds.biaffine_dependency predict \
    -d 0 \
    -c config.ini \
    -p exp/arabic/sud/biaff-dep \
    --data data/sud/sud-treebanks-v2.6/SUD_Arabic-PADT/ar_padt-sud-test.conllu \
    --pred exp/arabic/sud/biaff-dep.sud.pred.conllx