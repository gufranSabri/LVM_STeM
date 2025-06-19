# python main.py --device 6,7 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB --model-args c2d_type=vitb_tc-0
# python main.py --device 6,7 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_ST --model-args c2d_type=vitb_tc-1
# python main.py --device 6,7 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_ST_MSTCN --model-args c2d_type=vitb_mstcn-1
python main.py --device 6,7 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_TAPE --model-args c2d_type=vitb_tc-3
python main.py --device 6,7 --dataset phoenix2014 --work-dir /data/ahmed026/phoenix/ViTB_TAPE_MSTCN --model-args c2d_type=vitb_mstcn-3