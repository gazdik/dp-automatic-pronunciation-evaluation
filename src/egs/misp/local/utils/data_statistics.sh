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

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <data-dir>"
    echo "    e.g.: $0 data/train"
    exit 2;
fi

data_dir=$1

. ./path.sh

#########################
# Duration of audio files
#########################

duration_sec=0
wav_files=`cat $data_dir/wav.scp | grep -o -e '[^[:blank:]]*\.flac' -e '[^[:blank:]]*\.wav' -e '[^[:blank:]]*\.WAV' -e '[^[:blank:]]*\.FLAC'`
for f in $wav_files; do
  d=`soxi -D $f`
  duration_sec=`echo "$duration_sec+$d" | bc -l`
done 

#duration_sec=`wav-to-duration scp:${data_dir}/wav.scp ark,t:- | cut -f2 -d ' ' | awk '{ sum += $1 } END { print sum }'`
duration_min=`echo "scale=2; $duration_sec/60" | bc`
duration_hours=`echo "scale=2; $duration_min/60" | bc`

########################
# Number of speakers
########################

n_speakers=`cat $data_dir/spk2gender | wc -l`

########################
# Average duration
########################

avg_duration=`echo "scale=2; $duration_sec/60/$n_speakers" | bc`

echo "Total duration [hours]: $duration_hours"
echo "Total duration [minutes]: $duration_min"
echo "Number of speakers: $n_speakers"
echo "Average duration per speaker [minutes]: $avg_duration"

