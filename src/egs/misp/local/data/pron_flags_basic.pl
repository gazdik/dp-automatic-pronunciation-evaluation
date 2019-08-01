#!/usr/bin/env perl
# Copyright 2018 Peter Gazdik

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

# Compares a canonical and an actual transcriptions, creates mispronunciation
# flags for each phoneme, and also copies an insertion errors into the
# canonical transcription. Lengths of each utterances must be the same.

# use strict;
# use warnings;

$F_CORRECT = 0;
$F_SUBST = 1;
$F_DELET = 2;
$F_INSERT = 3;
$F_SILENT = 4;
$F_SILENT_MISMATCH = 5;
$F_UNKNOWN = 6;

$USAGE = <<"END_MSG";
Compares a canonical and an actual transcriptions, creates mispronunciation
flags for each phoneme, and also copies an insertion errors into the
canonical transcription. Lengths of each utterances must be the same.

Usage: pron_flags_extended.pl <in_canonic> <in_actual> <out_canonic> \
                           <out_actual> <out_flags>

Output flags:
    $F_CORRECT: Correct pronunciation
    $F_SUBST: Substitution error
    $F_DELET: Deletion error
    $F_INSERT: Insertion error
    $F_SILENT: Silent
    $F_SILENT_MISMATCH: Silent mismatch (only one phoneme is silent)
    $F_UNKNOWN: Unknown error
END_MSG

# Open input files
$canonic_in = shift @ARGV;
$actual_in = shift @ARGV;
$canonic_out = shift @ARGV;
$actual_out = shift @ARGV;
$flags_out = shift @ARGV;
if (!defined $canonic_in || !defined $actual_in || !defined $canonic_out ||
    !defined $actual_out || !defined $flags_out) {
    print STDERR $USAGE;
}
open(CANONIC_IN, $canonic_in) || die "Error opening file $canonic_in";
open(ACTUAL_IN, $actual_in) || die "Error opening file $actual_in";
open(CANONIC_OUT, ">$canonic_out") || die "Error opening file $canonic_out";
open(ACTUAL_OUT, ">$actual_out") || die "Error opening file $actual_out";
open(FLAGS_OUT, ">$flags_out") || die "Error opening file $flags_out";

print "create_pron_flags.pl: Processing the files $canonic_in and $actual_in\n";

# Read files simultaneously and print results to the output
while (($A_line = <ACTUAL_IN>) && ($C_line = <CANONIC_IN>)) {
    chomp($A_line); chomp($C_line);
    @A = split(" ", $A_line);
    @C = split(" ", $C_line);
    @Fout = ();
    @Aout = ();
    @Cout = ();
    
    # Check if transcriptions are correct.
    if ($A[0] ne $C[0]) {
        die "pron_flags_basic.pl: input text files aren't sorted";
    }
    if (@A != @C)  {
        $lenA = $#A + 1; $lenC = $#C + 1;
        die "pron_flags_basic.pl: transcriptions of utterance $A[0] have different lengths ($lenA \!\= $lenC)";
    }

    # Add utterance ID to each output
    push @Fout, $A[0]; push @Aout, $A[0]; push @Cout, $A[0];

    # Detect mispronunciation error and update outputs accordingly.
    for ($n = 1; $n < @A; $n++) {
        if ($C[$n] eq "sil" && $A[$n] eq "sil") {
            push @Fout, $F_SILENT;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
        elsif ($C[$n] eq "sil" && $A[$n] ne "sil") {
            push @Fout, $F_SILENT_MISMATCH;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
        elsif ($A[$n] eq $C[$n]) {
            push @Fout, $F_CORRECT;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
        elsif ($A[$n] eq "0") {
            push @Fout, $F_DELET;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
        elsif ($A[$n] =~ /.-./) {
            push @Fout, $F_INSERT;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
        else {
            push @Fout, $F_SUBST;
            push @Aout, $A[$n];
            push @Cout, $C[$n];
        }
    }
    print FLAGS_OUT join(" ", @Fout), "\n";
    print CANONIC_OUT join(" ", @Cout), "\n";
    print ACTUAL_OUT join(" ", @Aout), "\n";
}