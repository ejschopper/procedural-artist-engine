#!/bin/bash
cd /home/claude/artist_engine_v17_print
mkdir -p stage1
render() { python3 weathered.py --seed $1 --out stage1/s$1_150 --size-in 30 40 --dpi 150 > stage1/s$1.log 2>&1; }
render 2 & render 4 & wait
render 6 & render 7 & wait
for s in 2 4 6 7; do python3 -I upscale_detail.py stage1/s${s}_150 stage1/s${s}_300 >> stage1/s$s.log 2>&1; done
echo ALLDONE > stage1/done.txt
