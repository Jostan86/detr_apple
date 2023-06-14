#!/bin/bash
#SBATCH -J amodalFineTune						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/amodalFT/amodalFineTune-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/amodalFT/amodalFineTune-%a.err				  # name of error file for this submission script
#SBATCH -t 0-12:00:00                # time limit for job (HH:MM:SS)

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

source /nfs/hpc/share/browjost/detr_apple/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main2.py --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples/ --batch_size 2 --resume /nfs/hpc/share/browjost/detr_apple/weights/detr-r50_no-class-head.pth --epochs 50 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/amodalFT --dataset_file coco_apples_amodal
