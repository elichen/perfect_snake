#!/bin/bash
# Paired continuation experiment (2026-06-11), fires after sweep2's s404 audit.
# Both arms resume the 42.5% checkpoint (s202r best_eval_resume @146.3M, Adam moments
# intact) at constant lr 1e-5 (the LR band where s202r's wins crystallized), +120M steps,
# same env seed — a paired design that sidesteps the huge from-scratch basin variance.
# Arm A: continuation alone. Arm B: + tail-safety PBRS (policy-invariant shaping).
cd /Users/elichen/code/perfect_snake
RESUME=experiments/sweep_control_s202r_178111665825/best_eval_resume.pt
COMMON="--board-size 20 --timesteps 120000000 --num-envs 256 --horizon 256 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 20 --flood-fill --aux-flood-fill --gamma 0.999 --gae-lambda 0.9 --vf-clip-coef 1.0 --curriculum-prob 0.3 --head-centered --seed 777"
CONT="--resume-state $RESUME --resume-add-steps --override-resume-lr --lr 1e-5 --no-anneal-lr"

run_audit() {
  NAME=$1
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
}

echo "=== TRAIN cont_ctrl_s202r $(date) ==="
caffeinate -i .venv/bin/python -u train.py $COMMON $CONT \
  --exp-name cont_ctrl_s202r > experiments/cont_ctrl_s202r.log 2>&1
run_audit cont_ctrl_s202r

echo "=== TRAIN cont_pbrs_s202r $(date) ==="
caffeinate -i .venv/bin/python -u train.py $COMMON $CONT \
  --tail-safety-pbrs -0.3 --tail-safety-pbrs-min-fill 0.80 \
  --exp-name cont_pbrs_s202r > experiments/cont_pbrs_s202r.log 2>&1
run_audit cont_pbrs_s202r

echo "=== SWEEP3 COMPLETE $(date) ==="
