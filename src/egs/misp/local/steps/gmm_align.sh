#!/bin/bash

# Configuration section
nj=1
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
config= # name of config file.
splice_opts="--left-context=5 --right-context=5"
norm_means=true
norm_vars=false
delta_opts=
ctm_out=false
careful=false
text_fname="text"
beam=10
retry_beam=40
# End configuration section.

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [[ $# != 4 ]]; then
   echo "usage: local/pron/gmm_align.sh <lang-dir> <data-dir> <model-dir> <alignment-dir>"
   echo "e.g.: local/pron/gmm_align.sh data/lang data/train \\"
   echo "                                 exp/mono exp/train_force_ali"
   echo "main options (for others, see top of script file)"
   exit 1;
fi

lang_dir=$1
data_dir=$2
model_dir=$3
ali_dir=$4

if [[ -e ${ali_dir} ]]; then
    echo "Skipping aligning (already aligned)"
    exit 0
fi

# Create directories
mkdir -p ${ali_dir}/log

oov_sym=`cat ${lang_dir}/oov.int` || exit 1;
cp ${lang_dir}/phones.txt ${ali_dir} || exit 1;
cp ${model_dir}/{final.mdl,tree} ${ali_dir} || exit 1;

# Create features pipeline
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --norm-means=$norm_means "
feats+="--utt2spk=ark:$data_dir/utt2spk "
feats+="scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp ark:- |"
feats+="add-deltas $delta_opts ark:- ark:- |"


# Compile FST for each utterance
${cmd} ${ali_dir}/log/compile_graphs.log \
    compile-train-graphs --read-disambig-syms=${lang_dir}/phones/disambig.int \
    ${model_dir}/tree ${model_dir}/final.mdl ${lang_dir}/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang_dir/words.txt < $data_dir/$text_fname |" \
    "ark:|gzip -c > $ali_dir/fsts.gz" || exit 1;

mdl="gmm-boost-silence --boost=${boost_silence} `cat ${lang_dir}/phones/optional_silence.csl` ${model_dir}/final.mdl - |"
# Force align all utterances
${cmd} ${ali_dir}/log/align.log \
    gmm-align-compiled ${scale_opts} --beam=${beam} --retry-beam=${retry_beam} \
    --careful=${careful} "${mdl}" "ark:gunzip -c ${ali_dir}/fsts.gz |" \
    "${feats}" "ark,t:| gzip -c > ${ali_dir}/ali.gz" || exit 1;

