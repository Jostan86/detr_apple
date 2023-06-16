#!/bin/bash
#SBATCH -J dualHeadSeg						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/dualHeadSeg/dualHeadSeg-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/dualHeadSeg/dualHeadSeg-%a.err				  # name of error file for this submission script
#SBATCH --gres=gpu:1
#SBATCH -t 0-30:00:00                # time limit for job (HH:MM:SS)
#SBATCH --nodelist=dgx2-3

module load python3
source /nfs/hpc/share/browjost/detr_apple/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --masks --lr_drop 15 --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples/ --batch_size 2 --epochs 25 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/dualHeadSeg --dataset_file coco_apples_both_masks --frozen_weights /nfs/hpc/share/browjost/detr_apple/logdirs/amodalFT/checkpoint.pth
