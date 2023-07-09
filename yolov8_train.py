from ultralytics import YOLO

# Load a model
def main() :
    # model = YOLO("D:/yolo/06_22/weights/best.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch



    # Use the model
    model.train(data="d:/yolo/07_06.yaml", epochs=1000,imgsz=640,device=0,batch=16,name="d:/yolo/07_06",workers = 1)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("d:/data/ref7.jpg")  # predict on an image
    # success = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    main()