#!/bin/bash

# Copyright 2019 Peter Gazdik
#           2012 Vassil Panayotov
# Apache 2.0

. ./path.sh || exit 1

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <lang>";
  echo " Options:"
  echo "    <lang>        dataset language (voxforge_de, voxforge_it)"
  exit 1;
fi

lang=$1
data_dir=data/${lang}
conf_dir=conf/${lang}
locdata=${data_dir}/local
locdict=$locdata/dict

echo "=== Preparing the dictionary ..."

if [ ! -f $locdict/cmudict/cmudict.0.7a ]; then
  echo "--- Downloading CMU dictionary ..."
  mkdir -p $locdict
  svn co http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $locdict/cmudict || exit 1;
fi

echo "--- Striping stress and pronunciation variant markers from cmudict ..."
perl $locdict/cmudict/scripts/make_baseform.pl \
  ${conf_dir}/voxforge.dic /dev/stdout |\
  sed -e 's:^\([^\s(]\+\)([0-9]\+)\(\s\+\)\(.*\):\1\2\3:' > $locdict/dict-plain.txt

echo "--- Searching for OOV words ..."
awk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $locdict/dict-plain.txt $locdata/vocab-full.txt |\
  egrep -v '<.?s>' > $locdict/vocab-oov.txt

awk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $locdata/vocab-full.txt $locdict/dict-plain.txt |\
  egrep -v '<.?s>' > $locdict/lexicon-iv.txt

wc -l $locdict/vocab-oov.txt
wc -l $locdict/lexicon-iv.txt

if [[ "$(uname)" == "Darwin" ]]; then
  command -v greadlink >/dev/null 2>&1 || \
    { echo "Mac OS X detected and 'greadlink' not found - please install using macports or homebrew"; exit 1; }
  alias readlink=greadlink
fi

sequitur=$KALDI_ROOT/tools/sequitur-g2p
export PATH=$PATH:$sequitur/bin
export PYTHONPATH=$PYTHONPATH:`utils/make_absolute.sh $sequitur/lib/python*/site-packages`

if ! g2p=`which g2p.py` ; then
  echo "The Sequitur was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_sequitur.sh"
  exit 1
fi

if [ ! -f ${conf_dir}/voxforge.g2p_model ]; then
  echo "--- Training a trigram Sequitur G2P model ..."
  g2p.py --train ${conf_dir}/voxforge.dic --devel 5% --write-model ${conf_dir}/voxforge.g2p_model_1
  g2p.py --train ${conf_dir}/voxforge.dic --devel 5% --ramp-up --model ${conf_dir}/voxforge.g2p_model_1 --write-model ${conf_dir}/voxforge.g2p_model_2
  g2p.py --train ${conf_dir}/voxforge.dic --devel 5% --ramp-up --model ${conf_dir}/voxforge.g2p_model_2 --write-model ${conf_dir}/voxforge.g2p_model
  rm ${conf_dir}/voxforge.g2p_model_*
fi

echo "--- Preparing pronunciations for OOV words ..."
g2p.py --model=${conf_dir}/voxforge.g2p_model --apply $locdict/vocab-oov.txt > $locdict/lexicon-oov.txt

cat $locdict/lexicon-oov.txt $locdict/lexicon-iv.txt |\
  sort > $locdict/lexicon.txt
rm $locdict/lexiconp.txt 2>/dev/null || true

echo "--- Prepare phone lists ..."
echo SIL > $locdict/silence_phones.txt
echo SIL > $locdict/optional_silence.txt
grep -v -w sil $locdict/lexicon.txt | \
  awk '{for(n=2;n<=NF;n++) { p[$n]=1; }} END{for(x in p) {print x}}' |\
  sort > $locdict/nonsilence_phones.txt

echo "--- Adding SIL to the lexicon ..."
echo -e "!SIL\tSIL" >> $locdict/lexicon.txt

# Some downstream scripts expect this file exists, even if empty
touch $locdict/extra_questions.txt

echo "*** Dictionary preparation finished!"
