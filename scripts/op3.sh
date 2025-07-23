# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB --model-args c2d_type=vitb_tc-0
# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_ST --model-args c2d_type=vitb_tc-1
# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_ST_MSTCN --model-args c2d_type=vitb_mstcn-1
# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_TAPE --model-args c2d_type=vitb_tc-3
# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_TAPE_MSTCN --model-args c2d_type=vitb_mstcn-3



# ython main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_MSTCN --model-args c2d_type=swins_mstcn-0





# python main.py --device 4,5 --dataset CSL-Daily --work-dir /data/ahmed026/csldaily/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3 --optimizer-args step=[25,30,35,40,45] gamma=0.5 base_lr=0.00005 --num-epoch 50

python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3
python main.py --device 4,5 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3

# python main.py --device 4,5 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinSLoRA_TAPE_MSTCN --model-args c2d_type=swinslora_mstcn-3
# python main.py --device 4,5 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinSLoRA_TAPE_MSTCN --model-args c2d_type=swinslora_mstcn-3



