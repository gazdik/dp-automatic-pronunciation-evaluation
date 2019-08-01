#!/bin/bash

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: local/data/extend_transcripts.sh <lang-dir> <data-dir>"
   echo "e.g.: local/pronunciation/extend_transcripts.sh data/lang data/train"
   echo "main options (for others, see top of script file)"
   exit 1;
fi

langdir=$1
datadir=$2
tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Create pronunciation flags and extend canonical transcripts by adding
# insertion errors to them
local/data/pron_flags_extended.pl $datadir/text_canonic $datadir/text_actual \
    $datadir/text_ext_canonic $datadir/text_ext_actual $datadir/text_ext_flags

# Prepare a symbol table for the actual transcription
cut -d' ' -f2- $datadir/text_ext_actual | tr ' ' '\n' | sort -u > \
    $tmpdir/phones_ext_actual.txt
local/data/create_symbol_table.pl --old-symtab $langdir/phones.txt \
    $tmpdir/phones_ext_actual.txt > $langdir/phones_ext_actual.txt

# Convert transcriptions into the integer representation
sym2int.pl -f 2- $langdir/phones_ext_actual.txt $datadir/text_ext_actual > \
    $datadir/text_ext_actual.int
sym2int.pl -f 2- $langdir/phones.txt $datadir/text_ext_canonic > \
    $datadir/text_ext_canonic.int
sym2int.pl -f 2- $langdir/phones.txt $datadir/text_canonic > \
    $datadir/text_canonic.int
sym2int.pl -f 2- $langdir/phones.txt $datadir/text > \
    $datadir/text.int