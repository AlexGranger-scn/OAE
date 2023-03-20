# OAE
## Code Structure
```
OAE
├── README.md
├── requirement.txt
├─speed-test
│   ├── binary_tree.sh
│   ├── model.py
│   └── tree.py
├── test
│   ├── model.py
│   ├── test_oae-b.py
│   ├── test_oae-b.sh
│   ├── test-oae.py
│   └── test-oae.sh
└── train
    ├── train_oae
    │   ├── dataset.py
    │   ├── main.py
    │   ├── model.py
    │   └── train_oae.sh
    └── train_oae-b
        ├── dataset.py
        ├── main.py
        ├── model.py
        └── train_oae-b.sh
```

## How to run on your device

### Environment setup

1. create an python environment

```
conda create -n oae python==3.8
```

2. install the packages

```
conda activate oae

pip install -r requirement.txt
```

### Train

OAE: run
```shell
sh OAE/train/train_oae/train_oae.sh
```
OAE-b: run
```shell
sh OAE/train/train_oae-b/train_oae-b.sh
```

### Test

OAE: test

```shell
sh OAE/test/test_oae.sh
```

OAE-b: run

```shell
sh OAE/test/test_oae-b.sh
```

### 

