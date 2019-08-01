#!/usr/bin/env bash

# Copyright 2018 Peter Gazdik
# Copyright 2012 Vassil Panayotov

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
lm_order=2
pos_dep_phones=true
# -- Configuration section --

echo "$0 $@"  # Print the command line arguments

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [[ $# -ne 2 ]]; then
  echo "Usage: local/data/isle/prepare_data.sh <lang> <corpus-path>"
  echo " Options:"
  echo "    <lang>        dataset language (voxforge_de, voxforge_it)"
  echo "    <corpus-dir>  path to the voxforge corpus"
  exit 1;
fi

lang=$1
corpus_root=$2
corpus_dir=$corpus_root/extracted

data_dir=data/${lang}
log_dir=${data_dir}/log
script_dir=`dirname $0`

if [[ ! -d ${corpus_dir} ]]; then
    ${script_dir}/getdata.sh ${corpus_root}
fi

# Map anonymous speakers to unique IDs
$cmd ${log_dir}/map_anonymous.log \
    ${script_dir}/map_anonymous.sh ${corpus_dir} || exit 1
# Prepare data
$cmd ${log_dir}/data_prep.log \
    ${script_dir}/data_prep.sh ${lang} ${corpus_dir} || exit 1
# Prepare LM and vocabulary using SRILM
$cmd ${log_dir}/prepare_lm.log \
    ${script_dir}/prepare_lm.sh --order ${lm_order} ${lang} || exit 1
# Prepare lexicon
$cmd ${log_dir}/prepare_dict.log \
    ${script_dir}/prepare_dict.sh ${lang} || exit 1

# Prepare data/lang and data/local/lang directories
$cmd ${log_dir}/prepare_lang.log \
    utils/prepare_lang.sh --position-dependent-phones $pos_dep_phones \
    ${data_dir}/local/dict '!SIL' ${data_dir}/local/lang \
    ${data_dir}/lang || exit 1

# Format data
$cmd ${log_dir}/format_data.log \
    ${script_dir}/format_data.sh ${lang} || exit 1
