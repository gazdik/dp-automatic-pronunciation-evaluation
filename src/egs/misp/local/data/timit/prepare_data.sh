#!/usr/bin/env bash

# Copyright 2018 Peter Gazdík

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
# -- Configuration section --

echo "$0 $@"  # Print the command line arguments

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [[ $# -ne 2 ]]; then
  echo "Usage: local/data/timit/prepare_data.sh <lang> <corpus-path>"
  echo " Options:"
  echo "    <lang>        language of dataset"
  echo "    <corpus-dir>  path to the isle corpus"
  exit 1;
fi

lang=$1
corpus=$2

data_dir=data/${lang}
log_dir=${data_dir}/log

$cmd ${log_dir}/data_prep.log local/data/timit/data_prep.sh \
    ${corpus} || exit 1
$cmd ${log_dir}/prepare_dict.log local/data/timit/prepare_dict.sh

# Caution below: we remove optional silence by setting "--sil-prob 0.0",
# in TIMIT the silence appears also as a word in the dictionary is scored.
$cmd ${log_dir}/prepare_lang.log utils/prepare_lang.sh \
    --sil-prob 0.0 --position-dependent-phones false \
    --num-sil-states 3 ${data_dir}/local/dict "sil" \
    ${data_dir}/local/lang_tmp ${data_dir}/lang

$cmd ${log_dir}/format_data.log local/data/timit/format_data.sh
