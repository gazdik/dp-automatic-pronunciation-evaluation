#!/usr/bin/env bash

# Author: Peter Gazdik

if [[ $# -lt 1 ]]; then
  echo "Usage: experiment.sh <stage> [<stage> [<stage>] ...]"
  echo " Options:"
  echo "    stage       run a specific stage "
  exit 1;
fi

IFS=" "
stages=(`echo $@`)
lang=isle # Default language
base_lang=

function multi_run {
    exp_dir=$1
    stage_from=$2
    stage_to=$3
    ac_model=$4
    feats_type=$5

    mv ${exp_dir} ${exp_dir}_0

    for i in {1..4}; do
        if [[ -d ${exp_dir}_${i} ]]; then
            continue
        fi

        if [[ -n ${base_lang} ]]; then
            ./run.sh --stage_from ${stage_from} --stage_to ${stage_to} \
                --base_lang ${base_lang} \
                --ac_model ${ac_model} --feats_type ${feats_type} ${lang}
        else
            ./run.sh --stage_from ${stage_from} --stage_to ${stage_to} \
                --ac_model ${ac_model} --feats_type ${feats_type} ${lang}
        fi
 
        mv ${exp_dir} ${exp_dir}_${i}
    done
}

regex_lang='^[a-z][a-z_-]+[a-z_]$'
regex_multi_lang='^[a-z][a-z_-]+,[a-z_-]+[a-z_]$'
regex_move_folder='^[a-z][a-z_-]+\+[a-z_-]+[a-z_]$'
regex_copy_folder='^[a-z][a-z_-]+\+\+[a-z_-]+[a-z_]$'


while [[ -n ${stages[@]} ]]; do

if [[ "${stages[0]}" =~ $regex_multi_lang ]]; then
    lang=`echo ${stages[0]} | cut -d ',' -f2`
    base_lang=`echo ${stages[0]} | cut -d ',' -f1`
    stages=("${stages[@]:1}")    
fi

if [[ "${stages[0]}" =~ $regex_lang ]]; then
    lang=${stages[0]}
    base_lang=
    stages=("${stages[@]:1}")    
fi

if [[ "${stages[0]}" =~ $regex_move_folder ]]; then
    src=`echo ${stages[0]} | cut -d '+' -f1`
    dst=`echo ${stages[0]} | cut -d '+' -f2`

    # Rename src to dst
    mv -i exp/${src} exp/${dst}

    stages=("${stages[@]:1}")    
fi

if [[ "${stages[0]}" =~ $regex_copy_folder ]]; then
    src=`echo ${stages[0]} | cut -d '+' -f1`
    dst=`echo ${stages[0]} | cut -d '+' -f3`

    # Rename src to dst
    mkdir -p exp/$dst
    cp -r exp/${src}/mono exp/${dst}/
    cp -r exp/${src}/mono_ali exp/${dst}/
    cp -r exp/${src}/tri1 exp/${dst}/

    stages=("${stages[@]:1}")    
fi

stage=1
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "      Prepare data and MFCC, fbank features       "
    echo "**************************************************"

    # Prepare data and train mono and tri1 GMM models
    ./run.sh --stage_from 1 --stage_to 3 ${lang}
fi

stage=2
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "     Train mono and triphone GMM-HMM model        "
    echo "**************************************************"

    # Prepare mono and tri1 GMM models
    ./run.sh --stage 4 --stage_to 5 ${lang}
fi

stage=3
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "            Train mono DNN-HMM model              "
    echo "**************************************************"

    if [[ -n $base_lang ]]; then
        ./run.sh --stage 6 --base_lang ${base_lang} ${lang}
    else
        ./run.sh --stage 6 ${lang}
    fi
fi

stage=4
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "            Train tri1 DNN-HMM model              "
    echo "**************************************************"

    if [[ -n $base_lang ]]; then
        ./run.sh --stage 6 --ac_model tri1 --base_lang ${base_lang} ${lang}
    else
        ./run.sh --stage 6 --ac_model tri1 ${lang}
    fi
fi

stage=5
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "     Train classifier of phonological features    "
    echo "**************************************************"

    ./run.sh --stage 7 ${lang}
fi

stage=10
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "              NN HMM CLassifier                   "
    echo "**************************************************"

    # Train and evaluate basic mispronunciation classifiers
    if [[ -e exp/$lang/mono_pdf ]]; then
        ./run.sh --stage_from 9 --stage_to 10 --feats_type pdf --ac_model mono ${lang}
    fi

    if [[ -e exp/$lang/tri1_pdf ]]; then
        ./run.sh --stage_from 9 --stage_to 10 --feats_type pdf --ac_model tri1 ${lang}
    fi
fi

stage=11
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "              LR GOP and AP Scores                "
    echo "**************************************************"

    # Train and evaluate basic mispronunciation classifiers
    ./run.sh --stage_from 13 --stage_to 14 --ac_model mono ${lang} 
    ./run.sh --stage_from 13 --stage_to 14 --ac_model tri1 ${lang}
fi

stage=51
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "   Training and evaluating multiple instances     "
    echo "           of NN Mono HMM classifier              "
    echo "**************************************************"

    exp_dir=exp/${lang}/misp_mono_pdf
    multi_run ${exp_dir} 9 10 mono pdf
fi

stage=52
if [[ ${stages[0]} -eq ${stage} ]]; then stages=("${stages[@]:1}")
    echo "**************************************************"
    echo "   Training and evaluating multiple instances     "
    echo "           of NN Tri1 HMM classifier              "
    echo "**************************************************"

    exp_dir=exp/${lang}/misp_tri1_pdf
    multi_run ${exp_dir} 9 10 tri1 pdf
fi

done
