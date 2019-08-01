KALDI_ROOT=$(realpath ~/kaldi)       # Absolute path to Kaldi root directory
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] \
  && echo >&2 "\$KALDI_ROOT is not specified in path.sh!" \
  && echo >&2 "Please enter an absolute path to Kaldi root directory: " \
  && read KALDI_ROOT

export KALDI_ROOT=$KALDI_ROOT
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/gopbin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
