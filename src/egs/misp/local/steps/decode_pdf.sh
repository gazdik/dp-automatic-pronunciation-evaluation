#!/usr/bin/env bash

# Copyright 2019 Peter Gazdik

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
nj=4
cmd='run.pl'

max_active=7000 # max-active
beam=15.0 # beam used
latbeam=7.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices

skip_scoring=false # whether to skip WER scoring
scoring_opts=

splice_opts='--left-context=5 --right-context=5'
norm_means=true
norm_vars=true
# -- Configuration section --

export CUDA_VISIBLE_DEVICES="" # Don't use CUDA since we don't have enough GPUs

echo "$0 $@"  # Print the command line for logging
. parse_options.sh || exit 1;

# TODO update usage
if [[ $# -ne 4 ]]; then
   echo "Usage: decode.sh [options] <data-dir> <graph-dir> <dnn-dir> <decode-dir>"
   echo " e.g.: decode.sh data/test exp/tri2b/graph exp/dnn_5a exp/dnn_5a/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data_dir=$1
graph_dir=$2
dnn_dir=$3
decode_dir=$4
sdata_dir=${data_dir}/split${nj};
log_dir=${decode_dir}/log

mkdir -p ${log_dir}
[[ -d ${sdata_dir} ]] || split_data.sh ${data_dir} ${nj} || exit 1;
echo ${nj} > ${decode_dir}/num_jobs

for f in ${graph_dir}/HCLG.fst ${data_dir}/feats.scp ${dnn_dir}/final.mdl; do
  [[ ! -f ${f} ]] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
feats+="ark,s,cs:apply-cmvn --norm-vars=${norm_vars} "
feats+="--norm-means=${norm_means} --utt2spk=ark:${sdata_dir}/JOB/utt2spk "
feats+="scp:$sdata_dir/JOB/cmvn.scp scp:$sdata_dir/JOB/feats.scp ark:- |"
feats+="splice-feats $splice_opts ark:- ark:- |"
feats+="local/pron/pdf_forward.py $dnn_dir/final.h5 $dnn_dir/priors.csv |"

${cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
    latgen-faster-mapped --max-active=${max_active} --beam=${beam} \
    --lattice-beam=${latbeam} --acoustic-scale=${acwt} --allow-partial=true \
    --word-symbol-table=${graph_dir}/words.txt ${dnn_dir}/final.mdl \
    ${graph_dir}/HCLG.fst "${feats}" "ark:|gzip -c > ${decode_dir}/lat.JOB.gz"

if ! ${skip_scoring}; then
    [[ ! -x local/score.sh ]] && \
        echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
    local/score.sh ${scoring_opts} --cmd "$cmd" ${data_dir} \
        ${graph_dir} ${decode_dir}
fi

