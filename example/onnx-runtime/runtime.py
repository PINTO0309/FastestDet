import cv2
import time
import numpy as np
import onnxruntime 

# sigmoid function
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# tanh function
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

# Data preprocessing
def preprocess(src_img, size):
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    output = output.transpose(2,0,1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype('float32')

# NMS
def nms(dets, thresh=0.45):
    # dets: N*M, N is number of bboxes; first 4 are (x1,y1,x2,y2), 5th is score
    # #thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # area per bbox
    order = scores.argsort()[::-1]  # sort by score descending
    keep = []  # indices of kept bboxes

    while order.size > 0:
        i = order[0]  # keep highest-score bbox each iteration
        keep.append(i)

        # Compute overlap between highest-score bbox and remaining bboxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute intersection area
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # IoU between highest-score bbox and others
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep bboxes with IoU <= threshold and continue
        inds = np.where(ovr <= thresh)[0]

        # Shift because order[0] was excluded in ovr
        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

# Face detection
def detection(session, img, input_width, input_height, thresh):
    pred = []

    # Original image size
    H, W, _ = img.shape

    # Preprocess: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # Model inference
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # Feature map transpose: CHW -> HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # Feature map size
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # Feature map post-processing
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # Parse bbox confidence
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            # Threshold filtering
            if score > thresh:
                # Bbox class
                cls_index = np.argmax(data[5:])
                # Bbox center offset
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # Bbox normalized size
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # Bbox normalized center
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))

if __name__ == '__main__':
    # Load image
    img = cv2.imread("3.jpg")
    # Model input size
    input_width, input_height = 352, 352
    # Load model
    session = onnxruntime.InferenceSession('FastestDet.onnx')
    # Detection
    start = time.perf_counter()
    bboxes = detection(session, img, input_width, input_height, 0.65)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time:%fms"%time)

    # Load label names
    names = []
    with open("coco.names", 'r') as f:
	    for line in f.readlines():
	        names.append(line.strip())
            
    print("=================box info===================")
    for b in bboxes:
        print(b)
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        # Draw bounding boxes
        cv2.rectangle(img, (x1,y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(img, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
	
    cv2.imwrite("result.jpg", img)

