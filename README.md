# OAE
## Code Structure
```
OAE
├── README.md
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
### Train
OAE: run
```shell
sh OAE/train/train_oae/train_oae.sh
```
OAE-b: run
```shell
sh OAE/train/train_oae-b/train_oae-b.sh
```