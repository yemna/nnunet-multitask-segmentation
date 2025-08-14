import time
import os

# ----------------------------------------------------------------------------
# Note on Initial Model Performance
# ----------------------------------------------------------------------------
# This script focuses on optimizing the inference speed of a previously trained
# nnUNetv2 model. The initial vanilla training of nnUNetv2 produced segmentation
# metrics (specifically, DSC for both lesions and the whole pancreas) that
# exceeded expectations. The purpose of this assessment is to maintain that
# high accuracy while significantly improving the prediction time.

# ----------------------------------------------------------------------------
# BASELINE: The original command without speed optimizations
# ----------------------------------------------------------------------------
print("🚀 Running BASELINE prediction...")
start_time = time.time()

# The standard nnUNetv2 prediction command, which includes Test-Time Augmentation (TTA) by default.
os.system("nnUNetv2_predict -i D:/nnunet_implementation_m31_assessment/data/validation_flat -o D:/nnunet_implementation_m31_assessment/predictions_baseline -d 777 -f all -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetMPlans")

baseline_time = time.time() - start_time
print(f"✅ BASELINE completed in: {baseline_time:.2f} seconds")

# ----------------------------------------------------------------------------
# OPTIMIZED: The same command with a speed optimization flag
# ----------------------------------------------------------------------------
print("\n🚀 Running OPTIMIZED prediction...")
start_time = time.time()

# The optimized command adds the '--disable_tta' flag to remove Test-Time Augmentation for faster inference.
os.system("nnUNetv2_predict -i D:/nnunet_implementation_m31_assessment/data/validation_flat -o D:/nnunet_implementation_m31_assessment/predictions_optimized -d 777 -f all -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetMPlans --disable_tta")

optimized_time = time.time() - start_time
print(f"✅ OPTIMIZED completed in: {optimized_time:.2f} seconds")

# ----------------------------------------------------------------------------
# RESULTS: Compare and report the performance
# ----------------------------------------------------------------------------
# Ensure the commands ran for a reasonable duration before calculating results.
if baseline_time > 5 and optimized_time > 5:
    speedup = baseline_time / optimized_time
    improvement = ((baseline_time - optimized_time) / baseline_time) * 100
    
    print(f"\n🏁 RESULTS:")
    print(f"Baseline Time:   {baseline_time:.2f}s (with TTA)")
    print(f"Optimized Time:  {optimized_time:.2f}s (without TTA)") 
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Improvement:     {improvement:.1f}%")
    
    # Check if the target speed improvement was met
    if improvement >= 10:
        print("✅ Target ≥10% speed improvement ACHIEVED!")
    else:
        print("⚠️ Target ≥10% improvement not achieved.")
    
    # Save the results to a text file
    with open("speed_results.txt", "w") as f:
        f.write(f"Baseline Time: {baseline_time:.2f}s\n")
        f.write(f"Optimized Time: {optimized_time:.2f}s\n") 
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Improvement: {improvement:.1f}%\n")
        
else:
    print("❌ Commands completed too quickly, likely indicating a failure.")
    print(f"Baseline Time: {baseline_time:.2f}s, Optimized Time: {optimized_time:.2f}s")