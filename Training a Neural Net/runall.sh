rm -f results.csv
echo "config,seconds,loss,accuracy" > results.cvs

# Base case
python3 cifarnet.py base

# Speed convergence
python3 cifarnet.py swish
python3 cifarnet.py batchnorm
python3 cifarnet.py schedule

# Push generalization
python3 cifarnet.py adamw
python3 cifarnet.py dropout
