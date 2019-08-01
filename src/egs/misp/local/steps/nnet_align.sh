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
ctm_out=false
careful=false
text_fname="text"
beam=10
retry_beam=10
# End configuration section.

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: local/pron/nnet_align.sh <lang-dir> <data-dir> <classifier-dir> <alignment-dir>"
   echo "e.g.: local/pron/nnet_align.sh data/lang data/train \\"
   echo "                                 exp/misp exp/train_force_ali"
   echo "main options (for others, see top of script file)"
   exit 1;
fi

lang_dir=$1
data_dir=$2
dnn_dir=$3
ali_dir=$4

# Create directories
tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p ${ali_dir}/log

oov_sym=`cat ${lang_dir}/oov.int` || exit 1;
cp ${lang_dir}/phones.txt ${ali_dir} || exit 1;
cp ${dnn_dir}/{final.mdl,tree,final.h5,priors.csv} ${ali_dir} || exit 1;

# Create features pipeline
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --norm-means=$norm_means " 
feats+="--utt2spk=ark:$data_dir/utt2spk "
feats+="scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp ark:- |"
feats+="splice-feats $splice_opts ark:- ark:- |"
feats+="python3 $dnn_dir/forward.py $dnn_dir/final.h5 "
feats+="$dnn_dir/priors.csv |"

# Compile FST for each utterance
${cmd} ${ali_dir}/log/compile_graphs.log \
    compile-train-graphs --read-disambig-syms=${lang_dir}/phones/disambig.int \
    ${dnn_dir}/tree ${dnn_dir}/final.mdl ${lang_dir}/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang_dir/words.txt < $data_dir/$text_fname |" \
    "ark:|gzip -c > $ali_dir/fsts.gz" || exit 1;

# Force align all utterances
beam=10
${cmd} ${ali_dir}/log/align.log \
    align-compiled-mapped ${scale_opts} --beam=${beam} --retry-beam=$[$beam * 4] \
    ${dnn_dir}/final.mdl "ark:gunzip -c $ali_dir/fsts.gz |" "$feats" \
    "ark,t:| gzip -c > $ali_dir/ali.gz" || exit 1;


