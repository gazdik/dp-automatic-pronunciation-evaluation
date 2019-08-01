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

if [[ $# -lt 8 ]]; then
    echo "Usage: $0 <lang-dir> <conf-dir> <data-train> <data-dev> <ali-train> \\"
    echo "          <ali-dev> <feats-model-dir> <exp-dir>"
    echo "    e.g.: $0 data/lang conf \\"
    echo "             data/train_fbank_tr90 data/train_fbank_cv10 \\"
    echo "             exp/mono_ali_ext_tr10 exp/mono_ali_ext_cv10 \\"
    echo "             exp/mono_pdf exp/misp_pdf"
    echo " Options:"
    echo "    TODO  "
    exit 1;
fi

lang_dir=$1
conf_dir=$2
data_tr_dir=$3
data_cv_dir=$4
ali_tr_dir=$5
ali_cv_dir=$6
feats_model_dir=$7
exp_dir=$8

log_dir=${exp_dir}/log

mkdir -p ${exp_dir}
cp ${feats_model_dir}/final.h5 ${exp_dir}/feats.h5 # Copy feats' model
cp ${ali_tr_dir}/final.mdl ${exp_dir} # Copy GMM model
sym2int.pl -f 1 ${lang_dir}/phones.txt ${conf_dir}/phn_sil_to_idx.txt \
    > ${exp_dir}/phn_sil_to_idx.int

flags_tr="${data_tr_dir}/text_ext_flags"
flags_cv="${data_cv_dir}/text_ext_flags"

phones_tr="ark:ali-to-phones --per-frame ${ali_tr_dir}/final.mdl "
phones_tr+="\"ark:gunzip -c ${ali_tr_dir}/ali*.gz|\" ark:- |"
phones_cv="ark:ali-to-phones --per-frame ${ali_cv_dir}/final.mdl "
phones_cv+="\"ark:gunzip -c ${ali_cv_dir}/ali*.gz|\" ark:- |"

# Calculate mean and variance
apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} \
    --utt2spk=ark:"cat ${data_tr_dir}/utt2spk ${data_cv_dir}/utt2spk |" \
    scp:"cat ${data_tr_dir}/cmvn.scp ${data_cv_dir}/cmvn.scp |" \
    scp:"cat ${data_tr_dir}/feats.scp ${data_cv_dir}/feats.scp |" ark:- | \
    splice-feats ${splice_opts} ark:- ark:- | \
    ${feats_model_dir}/forward.py ${feats_model_dir}/final.h5 | \
    local/utils/cmvn_fit.py ${exp_dir}/cmvn.mdl

# Create files with features since we can't pass it
# as stream because of the "tensorflow deadlock"
apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} \
    --utt2spk=ark:${data_tr_dir}/utt2spk scp:${data_tr_dir}/cmvn.scp \
    scp:${data_tr_dir}/feats.scp ark:- | \
    splice-feats ${splice_opts} ark:- ark:- | \
    ${feats_model_dir}/forward.py ${feats_model_dir}/final.h5 | \
    local/utils/cmvn_transform.py ${exp_dir}/cmvn.mdl > \
    ${exp_dir}/feats_tr.ark

apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} \
    --utt2spk=ark:${data_cv_dir}/utt2spk scp:${data_cv_dir}/cmvn.scp \
    scp:${data_cv_dir}/feats.scp ark:- | \
    splice-feats ${splice_opts} ark:- ark:- | \
    ${feats_model_dir}/forward.py ${feats_model_dir}/final.h5 | \
    local/utils/cmvn_transform.py ${exp_dir}/cmvn.mdl > \
    ${exp_dir}/feats_cv.ark

# Train mispronunciation classifier
${cmd} ${log_dir}/misp_lstm_train.log local/pron/misp_lstm_train.py ${train_opts} \
    ${exp_dir}/feats_tr.ark ${exp_dir}/feats_cv.ark "${phones_tr}" "${phones_cv}" \
    ${flags_tr} ${flags_cv} ${exp_dir}

# Delete features
rm ${exp_dir}/feats_tr.ark
rm ${exp_dir}/feats_cv.ark
