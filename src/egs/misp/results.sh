#!/bin/bash
for x in exp/${1}/*/*decode*; do
  [ -d $x ] && echo $x | grep "${2:-.*}" >/dev/null \
  && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh;
done
for x in exp/${1}/*/*decode*; do
  [ -d $x ] && echo $x | grep "${2:-.*}" >/dev/null \
  && grep Sum $x/score_*/*.sys 2>/dev/null | utils/best_wer.sh; done
exit 0
