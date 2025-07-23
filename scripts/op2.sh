# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS --model-args c2d_type=swins_tc-0
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST --model-args c2d_type=swins_tc-1
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST_MSTCN --model-args c2d_type=swins_mstcn-1
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE --model-args c2d_type=swins_tc-3
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3



# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinB_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3


# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_STPE_MSTCN --model-args c2d_type=swins_mstcn-4
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_STPE_MSTCN --model-args c2d_type=vitb_mstcn-4



# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50


# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50

# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_TAPE_MSTCN_a1 --model-args c2d_type=vitb_mstcn-3


# python main.py --device 4,5,6,7 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50



# python main.py --device 6,7 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinT3D_TAPE_MSTCN --model-args c2d_type=swin3dtlora_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50





# python main.py --device 4,5 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/resSwinB_TAPE_MSTCN --model-args c2d_type=swinbres_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50



python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50
python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50
python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinB_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50


# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinB_TAPE_MSTCN --model-args c2d_type=swinblora_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50
# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinBLoRA_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50
# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinSLoRA_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50
# python main.py --device 2,3 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinTLoRA_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50