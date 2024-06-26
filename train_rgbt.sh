python train_simple.py \
  --img 640 \
  --batch-size 16 \
  --epochs 30 \
  --data data/kaist-rgbt-base.yaml \
  --cfg models/yolov5n_kaist-rgbt-improve.yaml \
  --weights yolov5n.pt \
  --workers 16 \
  --name yolov5n-backbone\
  --rgbt \
  --entity ttl9gg4 \
  --device 2



 

