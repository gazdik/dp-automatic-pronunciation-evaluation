#!/usr/bin/env bash

# Copyright 2018 Peter Gazdik

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# ++ Configuration section ++
cmd=run.pl
nj=4
norm_vars=true
norm_means=true
splice_opts='--left-context=5 --right-context=5'
train_opts=
# -- Configuration section --

echo "$0 $@"  # Print the command line for logging
. parse_options.sh || exit 1;

if [[ $# -lt 7 || $# -gt 9 ]]; then
    echo "Usage: $0 <lang-dir> <conf-dir> <pfeats-lang> \\"
    echo "          <data> <ali> <exp-dir> <eval-dir>"
    echo "    e.g.: $0 data/test exp/mono_ali_test exp/mono_pfeats exp/mono_pfeats/eval_test"
    echo " Options:"
    echo "    TODO  "
    exit 1;
fi

lang_dir=$1
conf_dir=$2
pfeats_lang=$3
data_dir=$4
ali_dir=$5
exp_dir=$6
eval_dir=$7

log_dir=${eval_dir}/log

mkdir -p ${eval_dir}
cp ${conf_dir}/phn_sil_to_idx.txt ${exp_dir}
cp ${lang_dir}/phones.txt ${exp_dir}
cp ${conf_dir}/pfeats_name_to_idx.txt ${exp_dir}
sym2int.pl -f 1 ${lang_dir}/phones.txt ${conf_dir}/phn_sil_to_idx.txt \
    > ${exp_dir}/phn_sil_to_idx.int

# Prepare feature streams
feats="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats+="--utt2spk=ark:${data_dir}/utt2spk scp:${data_dir}/cmvn.scp "
feats+="scp:${data_dir}/feats.scp ark:- |"
feats+="splice-feats ${splice_opts} ark:- ark:- |"
feats+="${exp_dir}/forward.py ${exp_dir}/final.h5 |"

phones+="ark:ali-to-phones --per-frame ${exp_dir}/final.mdl "
phones+="'ark:gunzip -c ${ali_dir}/ali*.gz |' ark:- |"

# Evaluate model
$cmd ${log_dir}/pfeats_eval.log local/pron/pfeats_eval.py ${train_opts} \
    "${feats}" "${phones}" $pfeats_lang ${exp_dir} ${eval_dir}

cat ${eval_dir}/accuracy.txt
