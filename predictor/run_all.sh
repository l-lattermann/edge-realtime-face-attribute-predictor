#!/bin/bash
# all runs for the report, one after the other
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
    echo "=== $NAME done  $(date +%H:%M)"
}

# main model, branch after row 9
run multitask_row9 --branch-row 9

# ablation a: one net per attribute
run single_expr   --task expr
run single_age    --task age
run single_gender --task gender

# ablation b: branch earlier, more resolution for expr
run multitask_row5 --branch-row 5

# row 4 is the only one that gives expr 28x28, see projektplan 5.2
run multitask_row4 --branch-row 4

echo "=== all runs finished  $(date +%H:%M)"
