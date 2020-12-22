import json
import cv2

with open("../yolact/data/coco/annotations/annotations/instances_val2017.json") as json_file:
    data = json.load(json_file)
    # print(type(data["images"][0]))
    # print((data["images"][2]))
    file_name = data["images"][2]['file_name']
    id = data["images"][2]['id']
    print(id)
    image = cv2.imread(file_name)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # print(data["annotations"])
    anns = data["annotations"]
    for ann in anns:
        if ann['image_id'] == id:
            seg = ann['segmentation'][0]
            print("segmentation size : " + str(len(ann['segmentation'][0])))
            # if len(ann['segmentation'][0]) > 20:
            #     continue
            for i in range(0, len(seg), 2):
                # print("(" + str(seg[i]) + "," + str(seg[i+1]) + ")")
                y = int(seg[i])-1
                x = int(seg[i+1])-1
                image[x][y] = 0

            # print(seg)
            # break

            # seg = int(seg)
            # for i in len(seg)/2:
            #     image[seg[i]][i+1] = 0
            # print("area : " + str(ann['area']))

            # print(ann)
    cv2.imshow("image", image)
    cv2.waitKey(0)