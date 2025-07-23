# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS --model-args c2d_type=swins_tc-0
# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST --model-args c2d_type=swins_tc-1
# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST_MSTCN --model-args c2d_type=swins_mstcn-1
# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE --model-args c2d_type=swins_tc-3
# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3



# python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3




# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3 --optimizer-args step=[20,30,40,50] --num-epoch 60
# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3 --optimizer-args step=[20,30,40,50] --num-epoch 60


# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3
# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinB_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3
# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3


# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinT_TAPE_MSTCN_LLMFT --model-args c2d_type=swint_mstcn-3 pretrained_w="/data/ahmed026/phoenixt/swinT_TAPE_MSTCN/_best_model.pt" --num-epoch 10




# python main.py --device 4,5 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinT3D_TAPE_MSTCN --model-args c2d_type=swin3dtlora_mstcn-3




# python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/resSwinB_TAPE_MSTCN --model-args c2d_type=swinbres_mstcn-3


# python main.py --device 0 --dataset phoenix2014-T --phase test --load-weights /data/ahmed026/phoenix/swinS_TAPE_MSTCN/_best_model.pt --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN_test



# python main.py --device 4,5 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/SSF_BsTf_hm --model-args c2d_type=swint_1dconv-3






python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3

python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinT_TAPE_MSTCN --model-args c2d_type=swint_mstcn-3

python main.py --device 0,1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinTLoRA_TAPE_MSTCN --model-args c2d_type=swintlora_mstcn-3

python main.py --device 0,1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinTLoRA_TAPE_MSTCN --model-args c2d_type=swintlora_mstcn-3





# python main.py --device 1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinSLoRA_TAPE_MSTCN --model-args c2d_type=swinslora_mstcn-3
# python main.py --device 1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinSLoRA_TAPE_MSTCN --model-args c2d_type=swinslora_mstcn-3
# python main.py --device 1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3
# python main.py --device 1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3





# python main.py --device 1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinB_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3
# python main.py --device 1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinB_TAPE_MSTCN --model-args c2d_type=swinb_mstcn-3
# python main.py --device 1 --dataset phoenix2014-T --work-dir /data/ahmed026/phoenixt/swinBLoRA_TAPE_MSTCN --model-args c2d_type=swinblora_mstcn-3
# python main.py --device 1 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinBLoRA_TAPE_MSTCN --model-args c2d_type=swinblora_mstcn-3


