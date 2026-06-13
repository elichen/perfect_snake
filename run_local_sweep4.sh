#!/bin/bash
# Continuation leg 2 (staged 2026-06-12): resume the 83.5% checkpoint
# (cont_ctrl_s202r best_eval_resume @~252M) for +120M more at constant lr 1e-5.
# Launch AFTER cont_pbrs_s202r's audit. If PBRS materially beat 83.5%, pass
# "pbrs" as $1 to carry --tail-safety-pbrs into this leg and resume the PBRS arm's
# checkpoint instead (set RESUME accordingly before launching).
cd /Users/elichen/code/perfect_snake
RESUME=${RESUME:-$(ls -t experiments/cont_ctrl_s202r_*/best_eval_resume.pt | head -1)}
EXTRA=""
NAME=cont2_ctrl_s202r
if [ "$1" = "pbrs" ]; then
  EXTRA="--tail-safety-pbrs -0.3 --tail-safety-pbrs-min-fill 0.80"
  NAME=cont2_pbrs_s202r
fi
COMMON="--board-size 20 --timesteps 120000000 --num-envs 256 --horizon 256 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 20 --flood-fill --aux-flood-fill --gamma 0.999 --gae-lambda 0.9 --vf-clip-coef 1.0 --curriculum-prob 0.3 --head-centered --seed 888"
CONT="--resume-state $RESUME --resume-add-steps --override-resume-lr --lr 1e-5 --no-anneal-lr"

echo "=== TRAIN $NAME resume=$RESUME $(date) ==="
caffeinate -i .venv/bin/python -u train.py $COMMON $CONT $EXTRA \
  --exp-name $NAME > experiments/${NAME}.log 2>&1

CKPT=$(ls -t experiments/${NAME}_*/best_eval.pt 2>/dev/null | head -1)
echo "=== AUDIT $NAME ckpt=$CKPT $(date) ==="
if [ -n "$CKPT" ]; then
  caffeinate -i .venv/bin/python -u -c "
import eval as E, json, math
s=E.evaluate_checkpoint(checkpoint_path='$CKPT',board_size=20,episodes=200,seed=200001,deterministic=True,device='mps',network_scale=2,flood_fill=True,aux_flood_fill=True,head_centered=True)
n=s['episodes']; w=s['wins']; p=w/n; z=1.96; d=1+z*z/n
c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
keys=['wins','win_rate','mean_score','median_score','steps_per_food','steps_per_food_by_fill','mean_win_steps','death_fill_buckets','death_reasons']
print('AUDIT $NAME', json.dumps({k:s.get(k) for k in keys}))
print('AUDIT $NAME wilson95 = %.1f%% to %.1f%%'%((c-h)*100,(c+h)*100))
" > "experiments/${NAME}_audit.log" 2>&1
fi
echo "=== SWEEP4 COMPLETE $(date) ==="
