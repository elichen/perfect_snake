#!/bin/bash
# Local sweep, phase 2 (2026-06-10). Completes the control distribution measurement.
# Arm 1: resume control s202 (died at 40M on Jun 3; resume from best_eval_resume.pt @35.3M) to 150M.
# Arm 2: fresh control s404. Both audited on 200 fresh seeds (200001-200200) for
# apples-to-apples vs baseline 21/200, tailsafe s101 0/200, topo 0/200.
# Rationale: with two mechanisms at 0/200, the control DISTRIBUTION is the decision
# hinge, so s404 replaces the originally planned tail-safety s303 arm.
cd /Users/elichen/code/perfect_snake
COMMON="--board-size 20 --timesteps 150000000 --num-envs 256 --horizon 256 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 20 --flood-fill --aux-flood-fill --gamma 0.999 --gae-lambda 0.9 --vf-clip-coef 1.0 --curriculum-prob 0.3 --head-centered"

run_audit() {
  NAME=$1
  CKPT=$(ls -t experiments/${NAME}_*/best_eval.pt 2>/dev/null | head -1)
  echo "=== AUDIT $NAME ckpt=$CKPT $(date) ==="
  if [ -n "$CKPT" ]; then
    caffeinate -i .venv/bin/python -u -c "
import eval as E, inspect, json, math
sig=inspect.signature(E.evaluate_checkpoint)
kw=dict(checkpoint_path='$CKPT',board_size=20,episodes=200,seed=200001,deterministic=True,device='mps',network_scale=2,flood_fill=True,aux_flood_fill=True)
if 'head_centered' in sig.parameters: kw['head_centered']=True
s=E.evaluate_checkpoint(**kw)
n=s['episodes']; w=s['wins']; p=w/n; z=1.96; d=1+z*z/n
c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
print('AUDIT $NAME', json.dumps({k:s.get(k) for k in ['wins','win_rate','mean_score','steps_per_food','mean_win_steps','death_fill_buckets','death_reasons']}))
print('AUDIT $NAME wilson95 = %.1f%% to %.1f%%'%((c-h)*100,(c+h)*100))
" > "experiments/${NAME}_audit.log" 2>&1
  fi
}

echo "=== TRAIN sweep_control_s202r (resume from 35.3M) $(date) ==="
caffeinate -i .venv/bin/python -u train.py $COMMON --seed 202 \
  --resume-state experiments/sweep_control_s202_178044620536/best_eval_resume.pt \
  --exp-name sweep_control_s202r > experiments/sweep_control_s202r.log 2>&1
run_audit sweep_control_s202r

echo "=== TRAIN sweep_control_s404 (fresh) $(date) ==="
caffeinate -i .venv/bin/python -u train.py $COMMON --seed 404 \
  --exp-name sweep_control_s404 > experiments/sweep_control_s404.log 2>&1
run_audit sweep_control_s404

echo "=== SWEEP2 COMPLETE $(date) ==="
