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

# Decoding options
max_active=7000 # max-active
beam=15.0 # beam used
latbeam=7.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices
# End configuration section.

# End configuration section.

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

# TODO update help
if [ $# != 4 ]; then
   echo "usage: local/pron/nnet_decode_best.sh <data-dir> <graph-dir> <nnet-dir> <decode-dir>"
#   echo "e.g.: local/pron/nnet_align.sh data/lang data/train \\"
#   echo "                                 exp/misp exp/train_force_ali"
   echo "main options (for others, see top of script file)"
   exit 1;
fi

datadir=$1
graphdir=$2
nnetdir=$3
decodedir=$4

# Create directories
mkdir -p ${decodedir}/log

cp ${nnetdir}/{final.mdl,tree} ${decodedir} || exit 1;

# Create features pipeline
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --norm-means=$norm_means " 
feats+="--utt2spk=ark:$datadir/utt2spk "
feats+="scp:$datadir/cmvn.scp scp:$datadir/feats.scp ark:- |"
feats+="splice-feats $splice_opts ark:- ark:- |"
feats+="python3 $nnetdir/forward.py $nnetdir/final.h5 "
feats+="$nnetdir/priors.csv |"

# Generate lattices
${cmd} ${decodedir}/log/decode.log \
    latgen-faster-mapped --max-active=${max_active} --beam=${beam} --lattice-beam=${latbeam} --acoustic-scale=${acwt} \
    --allow-partial=true --word-symbol-table=${graphdir}/words.txt ${nnetdir}/final.mdl ${graphdir}/HCLG.fst \
    "$feats" "ark:|gzip -c > $decodedir/lat.gz" || exit 1;

# Generate alignments
${cmd} ${decodedir}/log/1best.log \
    lattice-best-path --acoustic-scale=${acwt} \
    "ark:gunzip -c $decodedir/lat.gz|" ark:/dev/null \
    "ark:| gzip -c > $decodedir/ali.gz" || exit 1;

