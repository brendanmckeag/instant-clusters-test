#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import os
import time
import numpy as np
from datetime import timedelta

def setup_distributed():
    """Cloud-optimized distributed setup with Gloo fallback"""
    
    # Force Gloo backend for cloud stability
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_NET_GDR_LEVEL'] = '0'
    
    # Get distributed parameters
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    node_rank = global_rank // local_world_size
    
    print("="*70)
    print(f"🚀 RUNPOD CLOUD-STABLE MULTI-GPU DEMO")
    print("="*70)
    print(f"📡 Process {global_rank}: Initializing distributed training")
    print(f"   Node: {node_rank} | Local GPU: {local_rank} | Global Rank: {global_rank}")
    print(f"   World Size: {world_size} processes")
    print("-"*70)
    
    if global_rank == 0:
        print(f"⏳ Initializing with cloud-optimized backend...")
    
    # Try GLOO first (more stable for cloud)
    try:
        print(f"Process {global_rank}: Attempting Gloo backend...")
        dist.init_process_group(
            backend='gloo',
            timeout=timedelta(minutes=3)
        )
        backend_used = 'gloo'
        print(f"Process {global_rank}: ✅ Gloo backend successful")
        
    except Exception as e:
        print(f"Process {global_rank}: Gloo failed: {e}")
        print(f"Process {global_rank}: Trying NCCL with minimal settings...")
        
        # Ultra-conservative NCCL settings
        os.environ['NCCL_TIMEOUT'] = '300'  # 5 minutes
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=5)
        )
        backend_used = 'nccl'
        print(f"Process {global_rank}: ✅ NCCL backup successful")
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Display info
    if local_rank == 0:
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        print(f"🔥 Node {node_rank}: {local_world_size}× {gpu_name} ({gpu_memory:.1f}GB each)")
    
    if global_rank == 0:
        print(f"⚡ Backend: {backend_used.upper()} (Cloud optimized)")
        print("-"*70)
    
    return device, global_rank, world_size, local_rank, node_rank

class CloudStableModel(nn.Module):
    """Smaller model optimized for cloud distributed training"""
    def __init__(self, input_size=16, hidden_sizes=[64, 128, 64], output_size=8):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
        # Consistent initialization
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        if torch.distributed.get_rank() == 0:
            print(f"📊 Model: {total_params:,} parameters per GPU")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def safe_ddp_with_retries(model, device, local_rank, global_rank, max_retries=3):
    """DDP setup with retry logic for cloud environments"""
    
    for attempt in range(max_retries):
        try:
            print(f"Process {global_rank}: DDP setup attempt {attempt + 1}/{max_retries}")
            
            # Ensure all processes are ready
            dist.barrier()
            print(f"Process {global_rank}: Barrier passed")
            
            # Small delay to stagger initialization
            time.sleep(global_rank * 0.1)
            
            # Create DDP model with conservative settings
            ddp_model = DDP(
                model.to(device),
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=True,
                bucket_cap_mb=25,  # Smaller buckets for cloud
                gradient_as_bucket_view=True
            )
            
            print(f"Process {global_rank}: ✅ DDP setup successful")
            return ddp_model
            
        except Exception as e:
            print(f"Process {global_rank}: DDP attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

def create_dataset(num_samples=8000, input_size=16):
    """Smaller dataset for faster iteration"""
    torch.manual_seed(42)
    
    X = torch.randn(num_samples, input_size)
    true_weights = torch.randn(input_size, 8)
    logits = X @ true_weights + 0.1 * torch.randn(num_samples, 8)
    y = torch.softmax(logits, dim=1)
    
    return TensorDataset(X, y)

def main():
    device, global_rank, world_size, local_rank, node_rank = setup_distributed()
    
    # Create dataset
    dataset = create_dataset()
    if global_rank == 0:
        print(f"📈 Dataset: {len(dataset):,} samples")
    
    # Setup model
    if global_rank == 0:
        print(f"\n🔧 Setting up models across {world_size} GPUs...")
    
    model = CloudStableModel()
    
    # Safe DDP setup
    model = safe_ddp_with_retries(model, device, local_rank, global_rank)
    
    if global_rank == 0:
        print(f"✅ All models ready with DDP")
    
    # Training setup
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if global_rank == 0:
        print(f"📊 Training Configuration:")
        print(f"   Batch size per GPU: 64")
        print(f"   Effective batch size: {64 * world_size}")
        print(f"   Batches per epoch: {len(dataloader)}")
    
    # Training loop - 100 epochs
    total_epochs = 100
    all_losses = []
    
    if global_rank == 0:
        print(f"\n🚀 Starting training on {world_size} GPUs...")
    
    for epoch in range(total_epochs):
        if global_rank == 0:
            print(f"\n{'='*50}")
            print(f"🎯 EPOCH {epoch+1}/{total_epochs}")
            print(f"{'='*50}")
        
        sampler.set_epoch(epoch)
        epoch_losses = []
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            # Show progress from ALL GPUs for video demo
            if batch_idx % 8 == 0:  # Every 8 batches, show all GPUs
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(f"Node{node_rank}-GPU{local_rank} (Rank{global_rank}): {progress:5.1f}% | Loss: {loss.item():.4f}")
                time.sleep(0.05)  # Small delay so outputs don't overlap
        
        # Synchronize losses
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            loss_tensor = torch.tensor(avg_loss, device=device)
            
            try:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                global_loss = loss_tensor.item() / world_size
                all_losses.append(global_loss)
            except Exception as e:
                print(f"Process {global_rank}: All-reduce failed: {e}")
                global_loss = avg_loss
        
        epoch_time = time.time() - start_time
        
        if global_rank == 0:
            print(f"\n📊 EPOCH {epoch+1} COMPLETE:")
            print(f"   Global Loss: {global_loss:.6f}")
            print(f"   Time: {epoch_time:.1f}s")
            print(f"   GPUs Active: {world_size}")
            
            if len(all_losses) >= 2:
                improvement = all_losses[-2] - global_loss
                trend = "📉 Improving" if improvement > 0 else "📈 Adjusting"
                print(f"   Trend: {trend}")
        
        # 1 second pause between epochs
        time.sleep(1.0)
    
    # Simple verification
    if global_rank == 0:
        print(f"\n{'='*50}")
        print(f"🧪 VERIFICATION TEST")
        print(f"{'='*50}")
    
    # Basic connectivity test
    try:
        test_tensor = torch.tensor([1.0], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        if global_rank == 0:
            expected = world_size
            actual = test_tensor.item()
            print(f"Communication test: Expected {expected}, Got {actual}")
            success = abs(actual - expected) < 0.1
            print(f"Status: {'✅ PASS' if success else '❌ FAIL'}")
    except Exception as e:
        if global_rank == 0:
            print(f"❌ Communication test failed: {e}")
    
    # Final summary
    if global_rank == 0:
        print(f"\n{'='*50}")
        print(f"🎉 TRAINING COMPLETE!")
        print(f"{'='*50}")
        print(f"📊 Results:")
        print(f"   Epochs: {len(all_losses)}")
        if all_losses:
            print(f"   Final loss: {all_losses[-1]:.6f}")
            print(f"   Best loss: {min(all_losses):.6f}")
        print(f"   GPUs used: {world_size}")
        print(f"✅ Demo successful! 🚀")
        print(f"{'='*50}")
    
    # Cleanup
    try:
        dist.destroy_process_group()
    except:
        pass
    
    if global_rank == 0:
        print(f"🧹 Cleanup complete!")

if __name__ == "__main__":
    main()
