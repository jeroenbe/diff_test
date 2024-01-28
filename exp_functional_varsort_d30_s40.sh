
python -m exp_functional_varsort --run-count 30 -d 30 -s 40 -n 1000  --graph-type ER \
 -vs lin \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort

 python -m exp_functional_varsort --run-count 30 -d 30 -s 40 -n 1000  --graph-type ER \
 -vs lin-inv \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort




 python -m exp_functional_varsort --run-count 30 -d 30 -s 40 -n 1000  --graph-type ER \
 -vs exp \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort

python -m exp_functional_varsort --run-count 30 -d 30 -s 40 -n 1000  --graph-type ER \
 -vs exp-inv \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort




 python -m exp_functional_varsort --run-count 30 -d 30 -n 1000  --graph-type ER \
 -vs log \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort

 python -m exp_functional_varsort --run-count 30 -d 30 -n 1000  --graph-type ER \
 -vs log-inv \
 -m notears \
 -m dagma \
 -m ges \
 --group func_varsort
