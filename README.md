## build
```
g++ -std=c++11 -Wall -Wextra -O3 dnn.cpp -o train
```
## train mnist
```
./train --data_path ./data --learning_rate 1 --epochs 30 --batch_size 16
```
## train arithmetic
```
./train --learning_rate 0.1 --epochs 10000 --expr 2*6
```
