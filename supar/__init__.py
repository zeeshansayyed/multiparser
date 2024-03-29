# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser, MultiBiaffineDependencyParser,
                      BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      CRFNPDependencyParser, Parser,
                      VISemanticDependencyParser)

__all__ = ['BiaffineDependencyParser',
           'CRFNPDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'CRFConstituencyParser',
           'MultiBiaffineDependencyParser',
           'BiaffineSemanticDependencyParser',
           'VISemanticDependencyParser',
           'Parser']

__version__ = '1.0.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             MultiBiaffineDependencyParser,
                                             CRFNPDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             CRFConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

PRETRAINED = {
    'biaffine-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.biaffine.dependency.char.zip',
    'biaffine-dep-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.bert.zip',
    'biaffine-dep-bert-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.biaffine.dependency.bert.zip',
    'crfnp-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crfnp.dependency.char.zip',
    'crfnp-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crfnp.dependency.char.zip',
    'crf-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf2o.dependency.char.zip',
    'crf-con-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.constituency.char.zip',
    'crf-con-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.constituency.bert.zip',
    'crf-con-bert-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.constituency.bert.zip'
}
