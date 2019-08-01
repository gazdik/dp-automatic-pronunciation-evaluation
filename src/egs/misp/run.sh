#!/bin/bash

# Author: Peter Gazdik

# ++ Configuration section ++
stage=-1
stage_from=1
stage_to=42
base_lang=
num_leaves_tri1=2500
num_gauss_tri1=15000
ac_model=mono # mono, tri1, tri2, ...
feats_type=pdf # pdf, pfeats, pdf_pfeats
feats_type_1=pdf # pdf, pfeats
feats_type_2=pfeats # pdf, pfeats

feats_nj=10
train_nj=3
decode_nj=3

declare -A CORPUS_PATH=(
    ['isle']=/home/gazdik/datasets/ISLE
    ['isle_de']=/home/gazdik/datasets/ISLE
    ['isle_it']=/home/gazdik/datasets/ISLE
    ['timit']=/home/gazdik/datasets/TIMIT
    ['voxforge_de']=/home/gazdik/datasets/voxforge_de
    ['voxforge_it']=/home/gazdik/datasets/voxforge_it
)


# -- Configuration section --

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [[ ${stage} -ne -1 ]]; then
  stage_from=${stage}
  stage_to=${stage}
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: run.sh [--prev-lang <lang>] [--stage <stage>] [--stage_{from,to} <stage>] <lang>"
  echo " Options:"
  echo "    <dataset>      language of dataset (isle | timit | voxforge_de | voxforge_it )"
  echo "    --stage        run only a specific stage, it overwrites --stage_{from,to}"
  echo "    --stage_from   start execution from a specific stage"
  echo "    --stage_to     stop execution at a specific stage"
  exit 1;
fi

# you might not want to do this for interactive shells
set -e

lang=$1
data_dir=data/${lang}
conf_dir=conf/${lang}
exp_dir=exp/${lang}
[[ ! -z ${base_lang} ]] && base_exp_dir=exp/${base_lang}

stage=1
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "            Data & Lexicon & Language Preparation                   "
    echo ======================================================================

    local/data/${lang}/prepare_data.sh ${lang} ${CORPUS_PATH[$lang]}
fi

stage=2
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "      MFCC Feature Extraction & CMVN for Training and Test set      "
    echo ======================================================================

    mfcc_dir=${data_dir}/mfcc
    for x in train test; do
        steps/make_mfcc.sh --cmd "${train_cmd}" --nj ${feats_nj} \
            ${data_dir}/${x} ${mfcc_dir}/log/${x} ${mfcc_dir}
        steps/compute_cmvn_stats.sh ${data_dir}/${x} \
            ${mfcc_dir}/log/${x} ${mfcc_dir}
    done
fi

stage=3
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "      Fbank Feature Extraction & CMVN for Training and Test set     "
    echo ======================================================================

    fbank_dir=${data_dir}/fbank
    for x in train test; do
        cp -r ${data_dir}/${x} ${data_dir}/${x}_fbank
        steps/make_fbank.sh --cmd "${train_cmd}" --nj ${feats_nj} \
            ${data_dir}/${x}_fbank ${fbank_dir}/log/${x} ${fbank_dir}
        steps/compute_cmvn_stats.sh ${data_dir}/${x}_fbank \
            ${fbank_dir}/log/${x} ${fbank_dir}
    done
fi

stage=4
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "              Monophone GMM-HMM Training & Decoding                 "
    echo ======================================================================

    steps/train_mono.sh  --nj "${train_nj}" --cmd "${train_cmd}" \
        ${data_dir}/train ${data_dir}/lang ${exp_dir}/mono

    utils/mkgraph.sh ${data_dir}/lang_test \
        ${exp_dir}/mono ${exp_dir}/mono/graph

    rm -f local/score.sh; ln -rs local/data/${lang}/score.sh local/
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        ${exp_dir}/mono/graph ${data_dir}/test ${exp_dir}/mono/decode_test
fi

stage=5
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "         tri1: Deltas + Delta-Deltas Training & Decoding            "
    echo ======================================================================

    steps/align_si.sh --boost-silence 1.25 --nj "${train_nj}" \
        --cmd "${train_cmd}" ${data_dir}/train ${data_dir}/lang \
        ${exp_dir}/mono ${exp_dir}/mono_ali

    steps/train_deltas.sh --cmd "${train_cmd}" ${num_leaves_tri1} \
        ${num_gauss_tri1} ${data_dir}/train ${data_dir}/lang \
        ${exp_dir}/mono_ali ${exp_dir}/tri1

    utils/mkgraph.sh ${data_dir}/lang_test ${exp_dir}/tri1 \
        ${exp_dir}/tri1/graph

    rm -f local/score.sh; ln -rs local/data/${lang}/score.sh local/
    steps/decode.sh --nj "${decode_nj}" --cmd "${decode_cmd}" \
        ${exp_dir}/tri1/graph ${data_dir}/test ${exp_dir}/tri1/decode_test
fi

stage=6
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "                 DNN-HMM Training & Decoding                        "
    echo ======================================================================

    # Split training data into train and cross-validation sets
    utils/subset_data_dir_tr_cv.sh --cv-spk-list ${conf_dir}/cv_spk.list \
        ${data_dir}/train ${data_dir}/train_tr90 ${data_dir}/train_cv10
    utils/subset_data_dir_tr_cv.sh --cv-spk-list ${conf_dir}/cv_spk.list \
        ${data_dir}/train_fbank ${data_dir}/train_fbank_tr90 \
        ${data_dir}/train_fbank_cv10

    # Align data using the GMM model
    for x in cv10 tr90; do
        local/steps/gmm_align.sh ${data_dir}/lang \
            ${data_dir}/train_${x} ${exp_dir}/${ac_model} \
            ${exp_dir}/${ac_model}_ali_${x}
    done

    [[ ! -z ${base_exp_dir} ]] && base_pdf=${base_exp_dir}/${ac_model}_pdf

    local/steps/train_pdf.sh --cmd "$train_cmd" \
        ${data_dir}/train_fbank_tr90 \
        ${data_dir}/train_fbank_cv10 ${exp_dir}/${ac_model}_ali_tr90 \
        ${exp_dir}/${ac_model}_ali_cv10 ${exp_dir}/${ac_model}_pdf ${base_pdf}

    rm -f local/score.sh; ln -rs local/data/${lang}/score.sh local/
    local/steps/decode_pdf.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}/graph \
        ${exp_dir}/${ac_model}_pdf ${exp_dir}/${ac_model}_pdf/decode_test
fi

stage=7
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "              Phonological Features Classifier Training             "
    echo ======================================================================

    [[ ! -z ${base_exp_dir} ]] && base_pfeats=${base_exp_dir}/${ac_model}_pfeats

    local/steps/train_pfeats.sh --cmd "$train_cmd" \
        ${data_dir}/lang ${conf_dir} ${lang} \
        ${data_dir}/train_fbank_tr90 ${data_dir}/train_fbank_cv10 \
        ${exp_dir}/${ac_model}_ali_tr90 ${exp_dir}/${ac_model}_ali_cv10 \
        ${exp_dir}/${ac_model}_pfeats ${base_pfeats}

    # Force align test data
    local/steps/gmm_align.sh \
        --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_test

    local/steps/eval_pfeats.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${lang} ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_test \
        ${exp_dir}/${ac_model}_pfeats ${exp_dir}/${ac_model}_pfeats/eval_test
fi

stage=8
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "             PDF & Phonological Features Classifier Training        "
    echo ======================================================================

    [[ ! -z ${base_exp_dir} ]] && base_model=${base_exp_dir}/${ac_model}_pdf_pfeats

    local/steps/train_pdf_pfeats.sh --cmd "$train_cmd" \
        ${data_dir}/lang ${conf_dir} ${lang} \
        ${data_dir}/train_fbank_tr90 ${data_dir}/train_fbank_cv10 \
        ${exp_dir}/${ac_model}_ali_tr90 ${exp_dir}/${ac_model}_ali_cv10 \
        ${exp_dir}/${ac_model}_pdf_pfeats ${base_model}
fi

stage=9
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "           Mispronunciation Classifier Training                     "
    echo "                  (${feats_type} features)                          "
    echo ======================================================================

    # Force align with extended canonical transcription
    for x in "_cv10" "_tr90"; do
        local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
            --ctm-out true --norm-vars false --norm-means true \
            ${data_dir}/lang ${data_dir}/train${x} ${exp_dir}/${ac_model} \
            ${exp_dir}/${ac_model}_ali_ext${x}
    done

    # Train mispronunciation classifier
    local/steps/train_misp.sh --cmd "$train_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/train_fbank_tr90 ${data_dir}/train_fbank_cv10 \
        ${exp_dir}/${ac_model}_ali_ext_tr90 \
        ${exp_dir}/${ac_model}_ali_ext_cv10 \
        ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/misp_${ac_model}_${feats_type}
fi


stage=10
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "           Mispronunciation Classifier Evaluation                    "
    echo "                  (${feats_type} features)                           "
    echo "                     (on testing data)                               "
    echo =======================================================================

    # Force align with extended caonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/eval_misp.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/misp_${ac_model}_${feats_type} \
        ${exp_dir}/misp_${ac_model}_${feats_type}/eval_test
fi

stage=11
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "           LSTM Mispronunciation Classifier Training                "
    echo "                  (${feats_type} features)                          "
    echo ======================================================================

    # Force align with extended canonical transcription
    for x in "_cv10" "_tr90"; do
        local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
            --ctm-out true --norm-vars false --norm-means true \
            ${data_dir}/lang ${data_dir}/train${x} ${exp_dir}/${ac_model} \
            ${exp_dir}/${ac_model}_ali_ext${x}
    done

    # Train mispronunciation classifier
    local/steps/train_misp_lstm.sh --cmd "$train_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/train_fbank_tr90 ${data_dir}/train_fbank_cv10 \
        ${exp_dir}/${ac_model}_ali_ext_tr90 \
        ${exp_dir}/${ac_model}_ali_ext_cv10 \
        ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/misp_lstm_${ac_model}_${feats_type}
fi

stage=12
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "           LSTM Mispronunciation Classifier Evaluation               "
    echo "                  (${feats_type} features)                           "
    echo "                     (on testing data)                               "
    echo =======================================================================

    # Force align with extended canonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/eval_misp_lstm.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/misp_lstm_${ac_model}_${feats_type} \
        ${exp_dir}/misp_lstm_${ac_model}_${feats_type}/eval_test
fi

stage=13
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "                         LR GOP Score                                "
    echo =======================================================================

    # Force align with extended canonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/eval_gop_llr.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_${feats_type} ${exp_dir}/gop_llr_${feats_type}_${ac_model} \
        ${exp_dir}/gop_llr_${feats_type}_${ac_model}/eval_test
fi

stage=14
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "            GOP Score - Averaged Posteriors                          "
    echo =======================================================================

    # Force align with extended canonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/eval_gop_amll.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_${feats_type} ${exp_dir}/gop_amll_${feats_type}_${ac_model} \
        ${exp_dir}/gop_amll_${feats_type}_${ac_model}/eval_test
fi

stage=15
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "                         Standard GOP Score                          "
    echo =======================================================================

    # Force align with extended canonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    # Decode test data (free loop)
    local/steps/nnet_decode_best.sh --norm-vars true --norm-means true \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}/graph \
        ${exp_dir}/${ac_model}_pdf ${exp_dir}/${ac_model}_decode_test

    local/steps/eval_gop_std.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_decode_test ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/gop_std_${feats_type}_${ac_model} \
        ${exp_dir}/gop_std_${feats_type}_${ac_model}/eval_test
fi

stage=16
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "Standard GOP Score with modification according to the baseline article"
    echo =======================================================================

    local/steps/eval_gop_mod.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_decode_test ${exp_dir}/${ac_model}_${feats_type} \
        ${exp_dir}/gop_mod_${feats_type}_${ac_model} \
        ${exp_dir}/gop_mod_${feats_type}_${ac_model}/eval_test
fi

stage=17
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo ======================================================================
    echo "           Mispronunciation Classifier Training                     "
    echo "      (pasted ${feats_type_1} and ${feats_type_2} features)         "
    echo ======================================================================

    # Force align with extended canonical transcription
    for x in "_cv10" "_tr90"; do
        local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
            --ctm-out true --norm-vars false --norm-means true \
            ${data_dir}/lang ${data_dir}/train${x} ${exp_dir}/${ac_model} \
            ${exp_dir}/${ac_model}_ali_ext${x}
    done

    # Train mispronunciation classifier
    local/steps/train_2feats_misp.sh --cmd "$train_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/train_fbank_tr90 ${data_dir}/train_fbank_cv10 \
        ${exp_dir}/${ac_model}_ali_ext_tr90 \
        ${exp_dir}/${ac_model}_ali_ext_cv10 \
        ${exp_dir}/${ac_model}_${feats_type_1} \
        ${exp_dir}/${ac_model}_${feats_type_2} \
        ${exp_dir}/misp_2feats_${ac_model}_${feats_type_1}_${feats_type_2}
fi

stage=18
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "           Mispronunciation Classifier Evaluation                    "
    echo "      (pasted ${feats_type_1} and ${feats_type_2} features)         "
    echo "                     (on testing data)                               "
    echo =======================================================================

    # Force align with extended caonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/eval_2feats_misp.sh --cmd "$decode_cmd" ${data_dir}/lang ${conf_dir} \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/${ac_model}_${feats_type_1} \
        ${exp_dir}/${ac_model}_${feats_type_2} \
        ${exp_dir}/misp_2feats_${ac_model}_${feats_type_1}_${feats_type_2} \
        ${exp_dir}/misp_2feats_${ac_model}_${feats_type_1}_${feats_type_2}/eval_test
fi

stage=19
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
    echo =======================================================================
    echo "           Evaluation after averaging mispronunciation scores        "
    echo "                  (${feats_type} features)                           "
    echo "                     (on testing data)                               "
    echo =======================================================================

    # Force align with extended caonical transcription
    local/steps/gmm_align.sh --text-fname "text_ext_canonic" \
        --ctm-out true --norm-vars false --norm-means true \
        ${data_dir}/lang ${data_dir}/test ${exp_dir}/${ac_model} \
        ${exp_dir}/${ac_model}_ali_ext_test

    local/steps/avg_misp.sh --cmd "$decode_cmd" ${data_dir}/lang \
        ${data_dir}/test_fbank ${exp_dir}/${ac_model}_ali_ext_test \
        ${exp_dir}/misp_${ac_model}_${feats_type} \
        ${exp_dir}/misp_${ac_model}_${feats_type}_avg
fi

stage=42
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
  echo ============================================================================
  echo "              Getting results of speech recognition                       "
  echo ============================================================================

  bash results.sh '*'
fi

stage=43
if [[ ${stage} -ge ${stage_from} && ${stage} -le ${stage_to} ]]; then
  echo ============================================================================
  echo "              Getting results of mispronunciation detection               "
  echo ============================================================================

  tail -n +1 exp/${lang}/*/*/eer.txt
fi

echo =======================================================================
echo "Finished successfully on" `date`
echo =======================================================================

exit 0
