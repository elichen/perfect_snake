#!/bin/bash
# Local serial viability sweep on MPS (free). Each run: 150M steps, then a 200-fresh-seed
# audit (seeds 200001-200200) for apples-to-apples comparison vs the 10.5% baseline.
cd /Users/elichen/code/perfect_snake
COMMON="--board-size 20 --timesteps 150000000 --num-envs 256 --horizon 256 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 20 --flood-fill --aux-flood-fill --gamma 0.999 --gae-lambda 0.9 --vf-clip-coef 1.0 --curriculum-prob 0.3 --head-centered"

run_and_audit() {
  NAME=$1; shift
  echo "=== TRAIN $NAME $(date) ==="
  caffeinate -i .venv/bin/python -u train.py $COMMON "$@" --exp-name "$NAME" > "experiments/${NAME}.log" 2>&1
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

run_and_audit sweep_tailsafe015_s101 --tail-safety-penalty -0.15 --tail-safety-min-fill 0.80 --seed 101
run_and_audit sweep_control_s202 --seed 202
run_and_audit sweep_tailsafe015_s303 --tail-safety-penalty -0.15 --tail-safety-min-fill 0.80 --seed 303
echo "=== SWEEP COMPLETE $(date) ==="
