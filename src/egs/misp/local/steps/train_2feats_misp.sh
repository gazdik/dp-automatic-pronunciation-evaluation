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

if [[ $# -lt 9 ]]; then
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
feats_model_dir1=$7
feats_model_dir2=$8
exp_dir=$9

log_dir=${exp_dir}/log

mkdir -p ${exp_dir}
cp ${feats_model_dir1}/final.h5 ${exp_dir}/feats1.h5 # Copy feats' model
cp ${feats_model_dir2}/final.h5 ${exp_dir}/feats2.h5 # Copy feats' model
cp ${ali_tr_dir}/final.mdl ${exp_dir} # Copy GMM model
sym2int.pl -f 1 ${lang_dir}/phones.txt ${conf_dir}/phn_sil_to_idx.txt \
    > ${exp_dir}/phn_sil_to_idx.int

flags_tr="${data_tr_dir}/text_ext_flags"
flags_cv="${data_cv_dir}/text_ext_flags"

phones_ali_tr="ark:ali-to-phones --write-lengths ${ali_tr_dir}/final.mdl "
phones_ali_tr+="\\\"ark:gunzip -c ${ali_tr_dir}/ali*.gz|\\\" ark,t:- |"
phones_ali_cv="ark:ali-to-phones --write-lengths ${ali_cv_dir}/final.mdl "
phones_ali_cv+="\\\"ark:gunzip -c ${ali_cv_dir}/ali*.gz|\\\" ark,t:- |"

phones_tr="ark:ali-to-phones ${ali_tr_dir}/final.mdl "
phones_tr+="\"ark:gunzip -c ${ali_tr_dir}/ali*.gz |\" ark:- |"
phones_cv="ark:ali-to-phones ${ali_cv_dir}/final.mdl "
phones_cv+="\"ark:gunzip -c ${ali_cv_dir}/ali*.gz |\" ark:- |"

feats_tr="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats_tr+="--utt2spk=ark:${data_tr_dir}/utt2spk scp:${data_tr_dir}/cmvn.scp "
feats_tr+="scp:${data_tr_dir}/feats.scp ark:- |"
feats_tr+="splice-feats ${splice_opts} ark:- ark:- |"
feats_tr1="${feats_tr} ${feats_model_dir1}/forward.py ${feats_model_dir1}/final.h5 |"
feats_tr2="${feats_tr} ${feats_model_dir2}/forward.py ${feats_model_dir2}/final.h5 |"

feats_tr="ark:paste-feats \"${feats_tr1}\" \"${feats_tr2}\" ark:- |"
feats_tr+="local/utils/avg_feats.py \"${phones_ali_tr}\" |"

feats_cv="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats_cv+="--utt2spk=ark:${data_cv_dir}/utt2spk scp:${data_cv_dir}/cmvn.scp "
feats_cv+="scp:${data_cv_dir}/feats.scp ark:- |"
feats_cv+="splice-feats ${splice_opts} ark:- ark:- |"
feats_cv1="${feats_cv} ${feats_model_dir1}/forward.py ${feats_model_dir1}/final.h5 |"
feats_cv2="${feats_cv} ${feats_model_dir2}/forward.py ${feats_model_dir2}/final.h5 |"

feats_cv="ark:paste-feats \"${feats_cv1}\" \"${feats_cv2}\" ark:- |"
feats_cv+="local/utils/avg_feats.py \"${phones_ali_cv}\" |"

export CUDA_VISIBLE_DEVICES="" # Don't use CUDA

# Calculate mean and variance of averaged features
copy-feats "${feats_tr}" ark:- | \
    local/utils/cmvn_fit.py ${exp_dir}/cmvn.mdl
copy-feats "${feats_cv}" ark:- | \
    local/utils/cmvn_fit.py --model ${exp_dir}/cmvn.mdl ${exp_dir}/cmvn.mdl

# Apply transformations and save features into files
copy-feats "${feats_tr}" ark:- | \
    local/utils/cmvn_transform.py ${exp_dir}/cmvn.mdl > \
    ${exp_dir}/feats_tr.ark

copy-feats "${feats_cv}" ark:- | \
    local/utils/cmvn_transform.py ${exp_dir}/cmvn.mdl > \
    ${exp_dir}/feats_cv.ark

export CUDA_VISIBLE_DEVICES=0 # Use CUDA

# Check feature lengths if there isn't some mismatch
${cmd} ${log_dir}/check_lengths_tr.log local/utils/check_lengths.py \
    "${phones_tr}" ${exp_dir}/feats_tr.ark \
    --type_1 int_ark --type_2 mat_ark

${cmd} ${log_dir}/check_lengths_cv.log local/utils/check_lengths.py \
    "${phones_cv}" ${exp_dir}/feats_cv.ark \
    --type_1 int_ark --type_2 mat_ark
echo "Lengths checked"

# Train mispronunciation classifier
${cmd} ${log_dir}/misp_train.log local/pron/misp_train.py ${train_opts} \
    ${exp_dir}/feats_tr.ark ${exp_dir}/feats_cv.ark "${phones_tr}" "${phones_cv}" \
    ${flags_tr} ${flags_cv} ${exp_dir}

# Delete training features
rm ${exp_dir}/feats_tr.ark
rm ${exp_dir}/feats_cv.ark
