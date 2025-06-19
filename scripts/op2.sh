# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS --model-args c2d_type=swins_tc-0
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST --model-args c2d_type=swins_tc-1
python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_ST_MSTCN --model-args c2d_type=swins_mstcn-1
python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE --model-args c2d_type=swins_tc-3
# python main.py --device 2,3 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/swinS_TAPE_MSTCN --model-args c2d_type=swins_mstcn-3