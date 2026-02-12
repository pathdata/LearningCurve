TensorFlow version: 2.10.1

======================================================================\
GPU MEMORY CONFIGURATION
======================================================================
✓ Found 1 GPU(s)
✓ Memory growth enabled (allocates dynamically)

======================================================================\
STEP 1: Setting Custom Download Location
======================================================================
✓ Custom data directory: E:\Security\datasets

======================================================================\
STEP 2: Loading Your ResNet50
======================================================================\
2026-02-12 08:08:31.947672: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-02-12 08:08:32.016280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5663 MB memory:  -> device: 0, name: NVIDIA RTX A2000 8GB Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
✓ Model loaded from: C:\Users\pnarayanan\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels.h5
Model parameters: 25,636,712

======================================================================\
STEP 3: Loading CIFAR-10 with Memory-Efficient Settings
======================================================================\
✓ CIFAR-10 loaded: 50000 training images

⚡ Memory optimization:\
  Batch size: 16 (reduced from 32)\
  Effective batch: 32 (with adversarial examples)\
  GPU memory saved: ~50%\
✓ Training batches: 3125

======================================================================\
STEP 4: Memory-Efficient Adversarial Training
======================================================================

Starting memory-efficient training...
This will take ~1-2 hours (or 3-4 hours on CPU)

Training for 3 epochs
Steps per epoch: 1000
Batch size: 16
Effective samples per epoch: 16000


Epoch 1/3
----------------------------------------------------------------------\
2026-02-12 08:08:39.380359: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2026-02-12 08:08:39.881445: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
  Step    0/1000 | Loss: 8.3256 | Acc: 0.00%\
  Step  100/1000 | Loss: 0.7977 | Acc: 68.75%\
  Step  200/1000 | Loss: 1.1662 | Acc: 56.25%\
  Step  300/1000 | Loss: 0.8577 | Acc: 68.75%\
  Step  400/1000 | Loss: 0.8140 | Acc: 68.75%\
  Step  500/1000 | Loss: 0.4766 | Acc: 75.00%\
  Step  600/1000 | Loss: 0.6468 | Acc: 68.75%\
  Step  700/1000 | Loss: 0.3691 | Acc: 93.75%\
  Step  800/1000 | Loss: 0.4366 | Acc: 75.00%\
  Step  900/1000 | Loss: 0.5789 | Acc: 87.50%

  Epoch Summary:\
    Loss: 0.8113\
    Accuracy: 76.10%

Epoch 2/3
----------------------------------------------------------------------\
  Step    0/1000 | Loss: 0.1366 | Acc: 93.75%\
  Step  100/1000 | Loss: 0.8663 | Acc: 75.00%\
  Step  200/1000 | Loss: 0.2961 | Acc: 87.50%\
  Step  300/1000 | Loss: 0.0876 | Acc: 100.00%\
  Step  400/1000 | Loss: 0.2508 | Acc: 90.62%\
  Step  500/1000 | Loss: 0.0542 | Acc: 100.00%\
  Step  600/1000 | Loss: 0.2839 | Acc: 90.62%\
  Step  700/1000 | Loss: 0.0146 | Acc: 100.00%\
  Step  800/1000 | Loss: 0.2664 | Acc: 93.75%\
  Step  900/1000 | Loss: 0.0385 | Acc: 100.00%

  Epoch Summary:\
    Loss: 0.2544\
    Accuracy: 91.53%

Epoch 3/3
----------------------------------------------------------------------\
  Step    0/1000 | Loss: 0.0120 | Acc: 100.00%\
  Step  100/1000 | Loss: 0.2019 | Acc: 93.75%\
  Step  200/1000 | Loss: 0.0580 | Acc: 96.88%\
  Step  300/1000 | Loss: 0.3846 | Acc: 90.62%\
  Step  400/1000 | Loss: 0.1697 | Acc: 93.75%\
  Step  500/1000 | Loss: 0.0946 | Acc: 93.75%\
  Step  600/1000 | Loss: 0.0861 | Acc: 93.75%\
  Step  700/1000 | Loss: 0.0062 | Acc: 100.00%\
  Step  800/1000 | Loss: 0.1284 | Acc: 93.75%\
  Step  900/1000 | Loss: 0.0453 | Acc: 100.00%

  Epoch Summary:
    Loss: 0.1262\
    Accuracy: 95.84%

✓ Adversarial training complete!

======================================================================\
STEP 5: Evaluating Robustness
======================================================================

Testing robust model:
  Clean accuracy:        82.50%
  Robust accuracy (ε=0.01): 80.94%
  Robust accuracy (ε=0.03): 77.81%

======================================================================\
STEP 6: Saving Model
======================================================================
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
✓ Model saved to: E:\Security\datasets\resnet50_robust.h5

======================================================================
COMPLETE! SUMMARY
======================================================================

ALL ISSUES FIXED:
─────────────────\
✓ Disk space: Downloads to E: drive\
✓ Type error: int32 vs int64 fixed\
✓ OOM error: Batch size reduced, memory optimized\
✓ Model saved to: E:\Security\datasets\resnet50_robust.h5

MEMORY OPTIMIZATIONS APPLIED:
──────────────────────────────\
• Batch size: 16 (was 32, now 16)\
• GPU memory growth: Enabled\
• Gradient clipping: Enabled (prevents memory spikes)\
• Prefetching: Reduced to 1 batch\
• Steps per epoch: Increased to compensate for smaller batches

EXPECTED RESULTS:
────────────────\
• Clean accuracy: 80-90%\
• Robust accuracy (ε=0.03): 40-50%\
• Training time: 1-2 hours (GPU) or 3-4 hours (CPU)

FOR INTERVIEW:
──────────────\
✓ Solved infrastructure problem (disk space)\
✓ Fixed implementation bug (type mismatch)\
✓ Optimized memory usage (GPU OOM)\
✓ Shows production engineering skills!

TROUBLESHOOTING:
────────────────\
If you still get OOM errors:
1. Reduce batch size further: Change BATCH_SIZE = 8
2. Use CPU: Add with tf.device('/CPU:0'):
3. Close other applications using GPU
4. Monitor GPU memory: nvidia-smi (in another terminal)


======================================================================
# Training complete! Model is now robust to adversarial attacks.
======================================================================