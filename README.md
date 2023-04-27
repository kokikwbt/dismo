# DISMO

Python inplementation of DISMO, presented in "Modeling Dynamic Interaction over Tensor Streams".

# Installation

```sh
conda create -n dismo python=3.8
conda activate dismo
pip install -r requirements.txt
```

# Running examples

Following commnads start the streaming DISMO factorization after the initialization with the first 3-year tensor.

```bash
sh run_dismo_stream.sh ecommerce
sh run_dismo_stream.sh facilities
sh run_dismo_stream.sh sweets
sh run_dismo_stream.sh vod
```
