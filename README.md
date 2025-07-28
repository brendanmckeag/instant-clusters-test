Run this in the terminal on node 0 (master node): 
```
torchrun --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=10.65.0.2 \
  --master_port=29400 \
  distributed_demo_multigpu.py
```
Run the following on worker nodes (increase rank by 1 for each successive node)
```
torchrun --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.65.0.2 \
  --master_port=29400 \
  distributed_demo_multigpu.py
```
