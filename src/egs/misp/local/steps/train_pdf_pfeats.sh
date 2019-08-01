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

if [[ $# -lt 8 || $# -gt 9 ]]; then
    echo "Usage: $0 <lang-dir> <conf-dir> <pfeats-lang> \\"
    echo "          <data-train> <data-dev> <ali-train> \\"
    echo "          <ali-dev> <exp-dir> [<base-exp-dir>]"
    echo "    e.g.: $0 data/train_fbank_tr90 data/train_fbank_cv10 \\"
    echo "             exp/mono_ali_tr10 exp/mono_ali_cv10 exp/mono_pdf"
    echo " Options:"
    echo "    TODO  "
    exit 1;
fi

lang_dir=$1
conf_dir=$2
pfeats_lang=$3
data_tr_dir=$4
data_cv_dir=$5
ali_tr_dir=$6
ali_cv_dir=$7
exp_dir=$8

log_dir=${exp_dir}/log

if [[ $# -eq 9 ]]; then
    base_model=$9/final.h5
fi

mkdir -p ${exp_dir}
cp ${ali_tr_dir}/final.mdl ${exp_dir} # Copy GMM model
ln -frs local/pron/pdf_pfeats_forward.py ${exp_dir}/forward.py
sym2int.pl -f 1 ${lang_dir}/phones.txt ${conf_dir}/phn_sil_to_idx.txt \
    > ${exp_dir}/phn_sil_to_idx.int

# Prepare feature streams
feats_tr+="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats_tr+="--utt2spk=ark:${data_tr_dir}/utt2spk scp:${data_tr_dir}/cmvn.scp "
feats_tr+="scp:${data_tr_dir}/feats.scp ark:- |"
feats_tr+="splice-feats ${splice_opts} ark:- ark:- |"

feats_cv+="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats_cv+="--utt2spk=ark:${data_cv_dir}/utt2spk scp:${data_cv_dir}/cmvn.scp "
feats_cv+="scp:${data_cv_dir}/feats.scp ark:- |"
feats_cv+="splice-feats ${splice_opts} ark:- ark:- |"

pdf_tr+="ark:ali-to-pdf ${exp_dir}/final.mdl "
pdf_tr+="'ark:gunzip -c ${ali_tr_dir}/ali*.gz |' ark:- |"

pdf_cv+="ark:ali-to-pdf ${exp_dir}/final.mdl "
pdf_cv+="'ark:gunzip -c ${ali_cv_dir}/ali*.gz |' ark:- |"

phones_tr+="ark:ali-to-phones --per-frame ${exp_dir}/final.mdl "
phones_tr+="'ark:gunzip -c ${ali_tr_dir}/ali*.gz |' ark:- |"

phones_cv+="ark:ali-to-phones --per-frame ${exp_dir}/final.mdl "
phones_cv+="'ark:gunzip -c ${ali_cv_dir}/ali*.gz |' ark:- |"

# Train model
${cmd} ${log_dir}/pdf_pfeats_train.log local/pron/pdf_pfeats_train.py ${train_opts} \
    "${feats_tr}" "${feats_cv}" "${pdf_tr}" "${pdf_cv}" \
    "${phones_tr}" "${phones_cv}" ${pfeats_lang} \
    ${exp_dir} ${base_model}
