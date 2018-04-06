## build
```
g++ -std=c++11 -Wall -Wextra -O3 dnn.cpp -o train
```
## run
```
./train --data_path ./data --learning_rate 1 --epochs 30 --batch_size 16
```