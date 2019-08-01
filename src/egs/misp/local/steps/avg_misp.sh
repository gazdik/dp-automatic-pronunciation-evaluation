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

if [[ $# -lt 5 ]]; then
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
data_dir=$2
ali_dir=$3
exp_dir_prefix=$4
out_exp_dir=$5
eval_dir=${out_exp_dir}/eval_test

exp_dir_1=${exp_dir_prefix}_1

log_dir=${eval_dir}/log
mkdir -p ${eval_dir}
cp ${exp_dir_1}/phn_sil_to_idx.txt ${exp_dir_1}/phn_sil_to_idx.int ${out_exp_dir}
cp ${exp_dir_1}/final.h5 ${exp_dir_1}/feats.h5 ${exp_dir_1}/final.mdl ${out_exp_dir}
cp ${exp_dir_1}/cmvn.mdl ${out_exp_dir}

flags="${data_dir}/text_ext_flags"

# Evaluate performance of the mispronunciation classifier
${cmd} ${log_dir}/avg_score.log local/utils/avg_score.py \
    ${eval_dir}/score.ark ${exp_dir_prefix}_[0-9]*/eval_test/score.ark

# Create alignments in the text grid format
${cmd} ${log_dir}/align2ctm.log \
    ali-to-phones --ctm-output ${out_exp_dir}/final.mdl \
    "ark:gunzip -c ${ali_dir}/ali*.gz |" \
    ${eval_dir}/ali_phones.ctm

[[ ! -f ${data_dir}/utt2dur ]] && utils/data/get_utt2dur.sh ${data_dir}

#${cmd} ${log_dir}/ctm2textgrid.log local/utils/ctm2textgrid.py ${eval_dir}/ali_phones.ctm \
#    ${eval_dir}/score_textgrid ${lang_dir}/phones.txt \
#    ${data_dir}/utt2dur

## Add scores into alignments
#${cmd} ${log_dir}/score2textgrid.log local/utils/score2textgrid.py \
#    ${eval_dir}/score.txt ${eval_dir}/score_textgrid

# Calculate equal error rate
local/utils/eq_err_rate.py ${flags} ${eval_dir}/score.ark \
    --thr_start 0.0 --thr_stop 1.0 --trend dec > \
    ${eval_dir}/eer.txt

# Calculate equal error rate per phone
local/utils/eq_err_rate.py ${flags} ${eval_dir}/score.ark \
    --per_phone ${data_dir}/text_ext_canonic.int ${lang_dir}/phones.txt \
    --thr_start 0.0 --thr_stop 1.0 --trend dec > \
    ${eval_dir}/eer_per_phone.txt

# Print results
cat ${eval_dir}/eer*.txt
