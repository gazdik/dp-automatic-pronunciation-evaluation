#!/bin/bash

# Configuration section
nj=1
cmd=run.pl
# End configuration section.

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [[ $# != 2 ]]; then
   echo "usage: local/pron/gmm_align.sh <lang-dir> <data-dir> <model-dir> <alignment-dir>"
   echo "e.g.: local/pron/gmm_align.sh data/lang data/train \\"
   echo "                                 exp/mono exp/train_force_ali"
   echo "main options (for others, see top of script file)"
   exit 1;
fi

data_dir=$1
ali_dir=$2
log_dir=${ali_dir}/log

# Create alignments in the text grid format
${cmd} ${log_dir}/align2ctm.log \
    ali-to-phones --ctm-output ${ali_dir}/final.mdl \
    "ark:gunzip -c ${ali_dir}/ali*.gz |" \
    ${ali_dir}/ali_phones.ctm

[[ ! -f ${data_dir}/utt2dur ]] && utils/data/get_utt2dur.sh ${data_dir}

${cmd} ${log_dir}/ctm2textgrid.log local/utils/ctm2textgrid.py ${ali_dir}/ali_phones.ctm \
    ${ali_dir}/textgrid ${ali_dir}/phones.txt \
    ${data_dir}/utt2dur

