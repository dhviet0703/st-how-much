from ultralytics import YOLO, checks, hub
checks()

hub.login('1edc826aafc519fcea08a74a32c2c3880476ba148f')

model = YOLO('https://hub.ultralytics.com/models/WuKTrBJ8lTxiAGCCn1zH')
results = model.train()