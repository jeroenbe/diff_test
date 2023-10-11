python -m run --run-count 5 -d 20 -n 1000 --sem-type gauss \
 -m notears \
 -m dagma \
 --group exp1


 python -m run --run-count 5 -d 30 -n 1000 \
 -m dagma \
 --group exp1


 python -m run --run-count 5 -d 20 -n 1000 --sem-type mlp\
 -m dagma\
 --group exp1

 python -m run --run-count 5 -d 30 -n 1000 --sem-type mlp\
 -m dagma\
 --group exp1
