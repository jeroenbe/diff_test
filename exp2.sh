
python -m run --run-count 10 -d 20 -n 1000 --sem-type gauss --graph-type SF \
 -m notears \
 -m dagma \
 -m ges \
 --group exp2


 python -m run --run-count 10 -d 30 -n 1000 --graph-type SF \
 -m dagma \
 -m notears \
 -m ges \
 --group exp2


 python -m run --run-count 10 -d 20 -n 1000 --sem-type mlp --graph-type SF\
 -m dagma\
 -m notears \
 -m ges \
 --group exp2

 python -m run --run-count 10 -d 30 -n 1000 --sem-type mlp --graph-type SF\
 -m notears \
 -m ges \
 -m dagma\
 --group exp2
