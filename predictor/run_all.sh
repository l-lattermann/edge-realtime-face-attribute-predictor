#!/bin/bash
# every run the ablations need, primary model first
# start: tmux new-session -d -s train './run_all.sh'
# attach: tmux attach -t train

cd "$(dirname "$0")"
PYTHON=../.venv/bin/python

# skips runs that allready have a checkpoint, so i can restart it
run () {
    NAME=$1
    shift
    if [ -f "artifacts/$NAME.pt" ]; then
        echo "=== skip $NAME, checkpoint already there"
        return
    fi
    echo "=== $NAME  $(date +%H:%M)"
    $PYTHON train.py --run-name "$NAME" "$@" 2>&1 | tee "artifacts/${NAME}.log"
    $PYTHON predict.py --run-name "$NAME" --images val
    $PYTHON eval.py --run-name "$NAME" 2>&1 | tee "artifacts/${NAME}_eval.txt"
    echo "=== $NAME done  $(date +%H:%M)"
}

# the crop is square now, so the old ones do not match any more
rm -rf ../datasets/raf_db/cropped
$PYTHON prepare_rafdb.py

# the primary model, branch after row 9
run multitask_row9 --branch-row 9

# ablation (b): branch earlier, which shares less and keeps more resolution for expression
run multitask_row5 --branch-row 5
run multitask_row4 --branch-row 4

# ablation (a): one network per attribute, the ceiling the multi-task model is judged against
run single_expr   --task expr
run single_age    --task age
run single_gender --task gender

# the deployed model, branch after row 9: three precisions for the app
$PYTHON export.py --run-name multitask_row9 --branch-row 9

echo "=== all runs finished  $(date +%H:%M)"
