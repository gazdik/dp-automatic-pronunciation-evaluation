#!/bin/bash

# Derived from the TIMIT script.
# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#           2018   Brno University of Technology (Author: Peter Gazdik)
# Apache 2.0.

if [ $# -ne 2 ]; then
  echo "Usage: local/data/isle_data_prep.sh <isle-dir> <l1-lang>"
  echo " Options:"
  echo "    <l1-lang>      L1 language of speakers (isle | isle_it | isle_de )"
  echo "    <isle-dir>     Absolut path to a directory with ISLE corpus"
  exit 1;
fi

lang=$1
isledir=$2
datadir=data/$lang
dir=`pwd`/$datadir/local/data
lmdir=`pwd`/${datadir}/local/nist_lm{}
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

[ -f $conf/$lang/train_spk.list ] || error_exit "$PROG: Train-set speaker list not found.";
[ -f $conf/$lang/test_spk.list ] || error_exit "$PROG: Test-set speaker list not found.";

# First check if the train & test directories exist
if [ ! -d $isledir/ISLEDAT1 -o ! -d $isledir/ISLEDAT2 -o ! -d $isledir/ISLEDAT3 \
     -o ! -d $isledir/ISLEDAT4 ]; then
  echo "isle_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to ISLE directory."
  exit 1;
fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

cd $dir
for x in train test; do
  # First, find the list of audio files (use only BLOCK{D,E,F} utterances)
  find $isledir/ISLEDAT? -not \( -name 'BLOCKA*' -or -name 'BLOCKB*' \
    -or -name 'BLOCKC*' \) -name '*.WAV' | grep -f $conf/$lang/${x}_spk.list \
    > ${x}_wav.flist

  sed -e 's:.*/\(.*\)/.*/\(.*\).WAV$:\1_\2:i' ${x}_wav.flist \
    > $tmpdir/${x}_wav.uttids
  paste $tmpdir/${x}_wav.uttids ${x}_wav.flist | sort -k1,1 > ${x}_wav.scp

  cat ${x}_wav.scp | awk '{print $1}' > ${x}.uttids

  # Now, Convert the transcripts into our format (no normalization yet)
  # Get the actual transcripts: each line of the output contains an utterance
  # ID followed by the transcript.
  find $isledir/ISLEDAT?/SESS*/CLABS -not \( -name 'BLOCKA*' \
    -or -name    'BLOCKB*' -or -name 'BLOCKC*' \) -name '*.LAB' \
    | grep -f $conf/$lang/${x}_spk.list > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/.*/\(.*\).LAB$:\1_\2:i' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids

  # Get the canonical transcripts as well
  find $isledir/ISLEDAT?/SESS*/RLABS -not \( -name 'BLOCKA*' \
    -or -name    'BLOCKB*' -or -name 'BLOCKC*' \) -name '*.REF' \
    | grep -f $conf/$lang/${x}_spk.list > $tmpdir/${x}_canonic_phn.flist
  sed -e 's:.*/\(.*\)/.*/\(.*\).REF$:\1_\2:i' $tmpdir/${x}_canonic_phn.flist \
    > $tmpdir/${x}_canonic_phn.uttids

  for prefix in "" "_canonic"; do
    while read line; do
      [ -f $line ] || error_exit "Cannot find transcription file '$line'";
      # Remove additional information as deletions (0), insertions (-),
      #  intonations? (=) and comments (<>)
      cat "$line" | sed '/\/\/\//Q' | tr '[:upper:]' '[:lower:]' \
        | cut -f3 -d' ' | tr '\n' ' ' \
        | tr '0' -d | tr '-' ' ' | sed -e 's/<.*>//g'  \
        | perl -ape 's: *$:\n:;'
    done < $tmpdir/${x}${prefix}_phn.flist > $tmpdir/${x}${prefix}_phn.trans
    paste $tmpdir/${x}${prefix}_phn.uttids $tmpdir/${x}${prefix}_phn.trans \
      | sort -k1,1 | tr -d '\r' | tr '\t' ' ' > ${x}${prefix}.trans
    cat ${x}${prefix}.trans | sed -e 's/[[:blank:]]\+/ /g' \
      | sed -e 's/[[:blank:]]*$//g' \
      | sed -e 's/bckgrd/sil/g' | sed -e 's/sp/sil/g' \
      | sed -e 's/=eh/=nn/g' | sed -e 's/=ey/=nn/g' \
      | sed -e 's/=oh/=nn/g' | sed -e 's/=ow/=nn/g' \
      | sed -e 's/=p/=nn/g' | sed -e 's/=r/=nn/g' \
      | sed -e 's/=uw/=nn/g' | sed -e 's/=u/=nn/g' \
      | sed -e 's/\( sil\)\+/ sil/g' \
      | sort -k1,1 > ${x}${prefix}.text
  done

  # Extract also RAW CLAB version of transcriptions
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    # Remove additional information as deletions (0), insertions (-),
    #  intonations? (=) and comments (<>
    cat "$line" | sed '/\/\/\//Q' | tr '[:upper:]' '[:lower:]' \
      | cut -f3 -d' ' | tr '\n' ' ' \
      | sed -e 's/<.*>//g'  \
      | perl -ape 's: *$:\n:;'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_actual_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_actual_phn.trans \
    | sort -k1,1 | tr -d '\r' | tr '\t' ' ' > ${x}_actual.trans
  cat ${x}_actual.trans | sed -e 's/[[:blank:]]\+/ /g' \
    | sed -e 's/[[:blank:]]*$//g' \
    | sed -e 's/bckgrd/sil/g' | sed -e 's/sp/sil/g' \
    | sed -e 's/=eh/=nn/g' | sed -e 's/=ey/=nn/g' \
    | sed -e 's/=oh/=nn/g' | sed -e 's/=ow/=nn/g' \
    | sed -e 's/=p/=nn/g' | sed -e 's/=r/=nn/g' \
    | sed -e 's/=uw/=nn/g' | sed -e 's/=u/=nn/g' \
    | sed -e 's/\( sil\)\+/ sil/g' \
    | sort -k1,1 > ${x}_actual.text


  # Make the utt2spk and spk2utt files.
  cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;

  # Prepare gender mapping
  cat $conf/$lang/gender.list | grep -f $conf/$lang/${x}_spk.list | sort -k1,1 > $x.spk2gender

  # Prepare STM file for sclite:
  wav-to-duration --read-entire-file=true scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
  awk -v dur=${x}_dur.ark \
    'BEGIN{
     while(getline < dur) { durH[$1]=$2; }
     print ";; LABEL \"O\" \"Overall\" \"Overall\"";
     print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
     print ";; LABEL \"M\" \"Male\" \"Male speakers\"";
   }
   { wav=$1; spk=wav; sub(/_.*/,"",spk); $1=""; ref=$0;
     gender=(substr(spk,0,1) == "f" ? "F" : "M");
     printf("%s 1 %s 0.0 %f <O,%s> %s\n", wav, spk, durH[wav], gender, ref);
   }
  ' ${x}.text >${x}.stm || exit 1

  # Create dummy GLM file for sclite:
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > ${x}.glm

done

echo "Data preparation succeeded"
