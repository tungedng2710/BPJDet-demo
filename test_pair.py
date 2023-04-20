import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import yaml
import cv2
import time
from utils.datasets import LoadImages
from val import post_process_batch

colors_list = [
        # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
        # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

# bbox: [x1,y1,x2,y2,scores]
BODIES = [[7.44864e+02,  2.67153e+02,  1.11784e+03,  8.71960e+02,  8.72749e-01],
          [3.22921e+02,  2.66082e+02,  6.21655e+02,  8.64559e+02,  8.63467e-01],
          [1.11177e+03,  1.86403e+02,  1.48505e+03,  8.62382e+02,  7.93118e-01],
          [9.01986e+02,  2.14261e+02,  1.33223e+03,  8.59418e+02,  7.74857e-01],
          [4.29996e+00,  3.60969e+02,  1.39584e+02,  6.01113e+02,  7.55296e-01],
          [3.70525e+02,  2.03659e+02,  1.00164e+03,  8.64665e+02,  7.52747e-01],
          [4.19962e+02,  1.69850e+01,  1.09188e+03,  8.37484e+02,  7.40465e-01],
          [8.18489e+01,  2.75999e+02,  4.05355e+02,  8.72717e+02,  7.06716e-01],
          [8.98579e+02,  3.12349e+01,  1.36746e+03,  8.48553e+02,  6.73181e-01],
          [1.39707e+02,  3.25537e+01,  5.18816e+02,  8.55995e+02,  6.60087e-01],
          [7.74946e+02,  5.31480e+01,  1.17479e+03,  8.35652e+02,  6.50735e-01],
          [3.00945e+01,  1.67140e+02,  1.24311e+02,  3.88545e+02,  6.42930e-01],
          [3.08226e+02,  5.68134e+01,  7.61376e+02,  8.43467e+02,  5.37101e-01],
          [0,  1.73156e+02,  5.74467e+01,  3.77157e+02,  5.17830e-01],
          [0,  3.57018e+02,  8.79469e+01,  5.94591e+02,  5.01686e-01]]
FACES = [[1.28783e+03, 2.05912e+02, 1.35156e+03, 2.92807e+02, 9.08640e-01],
        [1.12907e+03, 2.44986e+02, 1.18922e+03, 3.27941e+02, 9.04746e-01],
        [1.17333e+03, 6.86364e+01, 1.23218e+03, 1.47326e+02, 8.93871e-01],
        [4.53531e+02, 3.04480e+02, 5.14845e+02, 3.91006e+02, 8.90762e-01],
        [9.13954e+02, 3.05884e+02, 9.73396e+02, 3.94822e+02, 8.87554e-01],
        [5.08196e+02, 8.26817e+01, 5.69427e+02, 1.70334e+02, 8.77525e-01],
        [2.94413e+02, 6.06961e+01, 3.55117e+02, 1.47382e+02, 8.73981e-01],
        [9.67267e+02, 8.55730e+01, 1.02510e+03, 1.61972e+02, 8.68917e-01],
        [2.24933e+02, 3.08461e+02, 2.88856e+02, 3.98316e+02, 8.36000e-01],
        [6.84076e+02, 2.24047e+02, 7.44462e+02, 3.20111e+02, 8.17462e-01],
        [7.38772e+02, 5.61135e+01, 7.99098e+02, 1.37259e+02, 7.90710e-01],
        [6.04432e+01, 1.83535e+02, 9.29361e+01, 2.22127e+02, 7.06487e-01],
        [6.33399e+01, 3.94583e+02, 9.85364e+01, 4.38815e+02, 6.42772e-01],
        [4.62215e+00, 3.92994e+02, 4.17826e+01, 4.38056e+02, 6.27756e-01],
        [4.18918e+00, 1.85754e+02, 3.17837e+01, 2.19552e+02, 4.97829e-01]]

def process_box(bodies, faces):
    for box in bodies:
        box.append(0)
        box.append((box[0]+box[2])/2)
        box.append((box[1]+box[3])/2)
    for box in faces:
        box.append(1)
        box.append((box[0]+box[2])/2)
        box.append((box[1]+box[3])/2)
    bodies = torch.Tensor([bodies])
    faces = torch.Tensor([faces])
    return bodies, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/JointBP_CityPersons_face.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.6, help='Matching IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--line-thick', type=int, default=2, help='thickness of lines')
    parser.add_argument('--counting', type=int, default=0, help='0 or 1, plot counting')

    args = parser.parse_args()

    with open("data/JointBP_CityPersons_face.yaml") as f:
        data = yaml.safe_load(f)  # load data dict
    data['conf_thres_part'] = 0.7  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = 0.5  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = 0.6  # whether a body-part in matched with one body bbox
    dataset = LoadImages("test_imgs/test1.jpg", img_size=1536, stride=64, auto=True)
    dataset_iter = iter(dataset)

    for index in range(len(dataset)):
        (single_path, img, im0, _) = next(dataset_iter)
        
        if '_res' in single_path or '_vis' in single_path:
            continue   
        img = torch.from_numpy(img)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        start = time.time()
        # --------------------------------------------------------------------------- #
        body_dets, part_dets = process_box(BODIES, FACES)
        bboxes, points, scores, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], body_dets, part_dets)
        # --------------------------------------------------------------------------- #
        end = time.time()
        print("Inference time: ", end - start)

        # args.line_thick = max(im0.shape[:2]) // 1280 + 3
        args.line_thick = max(im0.shape[:2]) // 1000 + 3
        instance_counting = 0
        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox
            color = colors_list[i%len(colors_list)]
            if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":  # data['num_offsets'] is 2
                f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                if data['part_type'] == "head" and f_score == 0:  # for the body-head pair, we must have a detected head
                    continue
                    
                instance_counting += 1               
                    
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)
                if f_score != 0:
                    [px1, py1, px2, py2] = f_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)

            if data['dataset'] == "BodyHands":  # data['num_offsets'] is 4
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)
                lh_score, lh_bbox = point[0][2], point[0][3:]  # left-hand part, bbox format [x1, y1, x2, y2]
                if lh_score != 0:
                    [px1, py1, px2, py2] = lh_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
                
                rh_score, rh_bbox = point[1][2], point[1][3:]  # right-hand part, bbox format [x1, y1, x2, y2]
                if rh_score != 0:
                    [px1, py1, px2, py2] = rh_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
        
        if args.counting:
            cv2.putText(im0, "Num:"+str(instance_counting), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(single_path[:-4]+"_res_%s.jpg"%(data['part_type']), im0)