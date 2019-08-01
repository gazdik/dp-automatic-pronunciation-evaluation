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
    echo "Usage: $0 <lang-dir> <conf-dir> <data-test> <ali-test> \\"
    echo "          <feats-model-dir> <exp-dir> <eval-dir>"
    echo "    e.g.: $0 data/lang conf data/test_fbank \\"
    echo "             exp/mono_ali_ext_test exp/mono_pdf exp/misp_pdf \\"
    echo "             exp/misp_pdf/eval_test \\"
    echo " Options:"
    echo "    TODO  "
    exit 1;
fi

lang_dir=$1
conf_dir=$2
data_dir=$3
ali_dir=$4
feats_model_dir1=$5
feats_model_dir2=$6
exp_dir=$7
eval_dir=$8

log_dir=${eval_dir}/log
mkdir -p ${eval_dir}
cp ${conf_dir}/phn_sil_to_idx.txt ${exp_dir}


flags="${data_dir}/text_ext_flags"

phones_ali="ark:ali-to-phones --write-lengths ${ali_dir}/final.mdl "
phones_ali+="\"ark:gunzip -c ${ali_dir}/ali*.gz|\" ark,t:- |"

phones="ark:ali-to-phones ${exp_dir}/final.mdl "
phones+="\"ark:gunzip -c ${ali_dir}/ali*.gz |\" ark:- |"

feats="ark:apply-cmvn --norm-vars=${norm_vars} --norm-means=${norm_means} "
feats+="--utt2spk=ark:${data_dir}/utt2spk scp:${data_dir}/cmvn.scp "
feats+="scp:${data_dir}/feats.scp ark:- |"
feats+="splice-feats ${splice_opts} ark:- ark:- |"

feats1="${feats} ${feats_model_dir1}/forward.py ${feats_model_dir1}/final.h5 |"
feats2="${feats} ${feats_model_dir2}/forward.py ${feats_model_dir2}/final.h5 |"

# Paste both features into one
export CUDA_VISIBLE_DEVICES="" # Don't use CUDA

paste-feats "${feats1}" "${feats2}" ark:- | \
    local/utils/avg_feats.py "${phones_ali}" | \
    local/utils/cmvn_transform.py ${exp_dir}/cmvn.mdl > \
    ${exp_dir}/feats_test.ark

export CUDA_VISIBLE_DEVICES=0 # Use CUDA

# Check feature lengths if there isn't some mismatch
${cmd} ${log_dir}/check_lengths_tr.log local/utils/check_lengths.py \
    "${phones}" ${exp_dir}/feats_test.ark \
    --type_1 int_ark --type_2 mat_ark

# Evaluate performance of the mispronunciation classifier
${cmd} ${log_dir}/misp_eval.log local/pron/misp_eval.py ${train_opts} \
    ${exp_dir}/feats_test.ark "${phones}" ${flags} ${exp_dir} ${eval_dir}

# Create alignments in the text grid format
${cmd} ${log_dir}/align2ctm.log \
    ali-to-phones --ctm-output ${exp_dir}/final.mdl \
    "ark:gunzip -c ${ali_dir}/ali*.gz |" \
    ${eval_dir}/ali_phones.ctm

[[ ! -f ${data_dir}/utt2dur ]] && utils/data/get_utt2dur.sh ${data_dir}

${cmd} ${log_dir}/ctm2textgrid.log local/utils/ctm2textgrid.py ${eval_dir}/ali_phones.ctm \
    ${eval_dir}/score_textgrid ${lang_dir}/phones.txt \
    ${data_dir}/utt2dur

# Add scores into alignments
${cmd} ${log_dir}/score2textgrid.log local/utils/score2textgrid.py \
    ${eval_dir}/score.txt ${eval_dir}/score_textgrid

# Calculate equal error rate
local/utils/eq_err_rate.py ${flags} ${eval_dir}/score.ark \
    --thr_start 0.0 --thr_stop 1.0 --trend dec > \
    ${eval_dir}/eer.txt

# Calculate equal error rate per phone
local/utils/eq_err_rate.py ${flags} ${eval_dir}/score.ark \
    --per_phone ${data_dir}/text_ext_canonic.int ${lang_dir}/phones.txt \
    --thr_start 0.0 --thr_stop 1.0 --trend dec > \
    ${eval_dir}/eer_per_phone.txt

# Plot ROC curve
local/utils/plot_roc.py ${flags} ${eval_dir}/score.ark dec \
    --thr_start 0.0 --thr_stop 1.0 --out_file ${eval_dir}/roc

# Plot ROC curves per phone
local/utils/plot_roc.py ${flags} ${eval_dir}/score.ark dec \
    --per_phone ${data_dir}/text_ext_canonic.int ${lang_dir}/phones.txt \
    --thr_start 0.0 --thr_stop 1.0 --out_file ${eval_dir}/roc_per_phone


# Delete features
rm ${exp_dir}/feats_test.ark

# Print results
cat ${eval_dir}/eer*.txt