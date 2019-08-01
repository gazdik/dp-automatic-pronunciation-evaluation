#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

. ./path.sh || exit 1

echo "=== Building a language model ..."


# Language model order
order=3

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <lang>";
  echo " Options:"
  echo "    <lang>        dataset language (voxforge_de, voxforge_it)"
  exit 1;
fi

lang=$1
data_dir=data/${lang}
locdata=${data_dir}/local
loctmp=$locdata/tmp

echo "--- Preparing a corpus from test and train transcripts ..."

# Prepare a LM training corpus from the transcripts _not_ in the test set
cut -f2- -d' ' < $locdata/test_trans.txt |\
  sed -e 's:[ ]\+: :g' | sort -u > $loctmp/test_utt.txt

# We are not removing the test utterances in the current version of the recipe
# because this messes up with some of the later stages - e.g. too many OOV
# words in tri2b_mmi
cut -f2- -d' ' < $locdata/train_trans.txt |\
   sed -e 's:[ ]\+: :g' |\
   sort -u > $loctmp/corpus.txt


loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
  else
    sdir=$KALDI_ROOT/tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi

ngram-count -order $order -write-vocab $locdata/vocab-full.txt -wbdiscount \
  -text $loctmp/corpus.txt -lm $locdata/lm.arpa

echo "*** Finished building the LM model!"
