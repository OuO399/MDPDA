#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS=../corpus/ambari.txt
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=ambari
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
VECTOR_SIZE=60
MAX_ITER=50
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10
if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

arr_dimension=(40)
arr_project=("ant_1.5_1.6" "ant_1.6_1.7" "camel_1.4_1.6" "ivy_1.4_2.0" "jEdit_4.0_4.1" "jEdit_4.1_4.2" "jEdit_4.2_4.3" "poi_2.0_2.5" "synapse_1.0_1.1" "synapse_1.1_1.2" "xalan_2.4_2.5" "xerces_1.2_1.3" "log4j_1.0_1.1")
for VECTOR_SIZE in ${arr_dimension[@]};do
    for SAVE_FILE in ${arr_project[@]};do
        CORPUS=../vocal_vec/$SAVE_FILE.txt
        echo
        echo "$ ../GloVe/$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
        ../GloVe/$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
        echo "$ ../GloVe/$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
        ../GloVe/$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
        echo "$ ../GloVe/$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
        ../GloVe/$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
        echo "$ ../GloVe/$BUILDDIR/glove -save-file ../GloVe/models/project/$VECTOR_SIZE/$SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
        ../GloVe/$BUILDDIR/glove -save-file ../GloVe/models/project/$VECTOR_SIZE/$SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
        if [ "$CORPUS" = 'text8' ]; then
           if [ "$1" = 'matlab' ]; then
               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
           elif [ "$1" = 'octave' ]; then
               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
           else
               echo "$ $PYTHON eval/python/evaluate.py"
               $PYTHON eval/python/evaluate.py
           fi
        fi
    done
done