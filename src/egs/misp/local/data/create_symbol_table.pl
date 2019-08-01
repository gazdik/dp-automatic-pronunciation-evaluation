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

# Creates an FST-like symbol table from the given list of symbols or
# optionaly reuses an existing symbol table.

# Process optional command options
for ($x = 0; $x < 1; $x++) {
    if ($ARGV[0] eq "--old-symtab") {
        shift @ARGV;
        $old_symtab = shift @ARGV;
        if ($old_symtab !~ /^.+\.txt$/ || $old_symtab eq "") {
            die "The --old-symtab option requires an argument.";
        }
    }
}

# Read the existing symbol table if it's given
if (defined $old_symtab) {
    open(F, $old_symtab) || die "Error opening symbol table file $old_symtab";
    while(<F>) {
        @A = split(" ", $_);
        @A == 2 || die "Bad line in symbol table file: $_";
        $sym2int{$A[0]} = $A[1] + 0;
    }
}

# Fill the symbol table
$symbols = shift @ARGV;
if (!defined $symbols) {
    print STDERR "Usage: create_symbol_table.pl [options] <symbols> > <symbol_table>\n" .
        "options: [--old-symtab <old_symtab>] \n";
}
open(F, $symbols) || die "Error opening file $symbols.";

$idx = keys %sym2int;
while (<F>) {
    chomp($_);
    $sym2int{$_} = $idx++;
}
if ($idx == 0) {
    die "There are no symbols in input file.";
}

# Write the symbol table to the output
while (($sym, $val) = each(%sym2int)) {
    print $sym, " ", $val, "\n"
}
