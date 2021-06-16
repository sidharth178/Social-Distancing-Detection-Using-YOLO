#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries

# In[10]:


import cv2
import sys
import numpy as np
from utils import transparentOverlay1, dst_circle, get_bounding_box, int_circle


# ## 2. Constants

# In[ ]:


# FLAGS
WRITE_VIDEO = True
SHOW_OUTPUT = True


# ## 3. Model Building

# In[53]:


def main(video_source):

    print("[INFO] loading YOLO ...")
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # determine only the *output* layer names that we need from YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    print("[INFO] start video streaming ...")
    # initialize the video stream
    cap = cv2.VideoCapture(video_source)

    if WRITE_VIDEO:
        # output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

        
    i = 0
    # loop over the frames from the video stream
    while (cap.isOpened()):
        
        # read the next frame from the file
        ret, frame = cap.read()
        
        # If the frame was not grabbed,then we have reached the end of the stream
        if not ret:
            break
            
        i += 1
        if not (i % 3 == 0): continue
        # grab the dimensions of the frame
        height, width, channels = frame.shape
        
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        boxes, confidences, class_id = get_bounding_box(outs, height, width)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
       
        circles = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]                                
                cx, cy = x + w // 2, y + h
                # draw a vertical line from the center of circle to center of boundary box 
                frame = cv2.line(frame, (cx, cy), (cx, cy - h // 2), (0, 255, 0), 2)
                # draw circle around the people
                frame = cv2.circle(frame, (cx, cy - h // 2), 5, (255, 20, 200), -1)
                circles.append([cx, cy - h // 2, h])

        int_circles_list = []
        indexes = []
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, len(circles)):
                x2, y2, r2 = circles[j]
                if int_circle(x1, y1, x2, y2, r1 // 2, r2 // 2) >= 0 and abs(y1 - y2) < r1 // 4:
                    indexes.append(i)
                    indexes.append(j)

                    int_circles_list.append([x1, y1, r1])
                    int_circles_list.append([x2, y2, r2])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            rows, cols, _ = frame.shape
            for i in range(len(circles)):
                x, y, r = circles[i]

                if i in indexes:
                    color = (0, 0, 255)
                else:
                    color = (0, 200, 20)
                scale = (r) / 100
                
                transparentOverlay1(frame, dst_circle, (x, y - 5), alphaVal=110, color=color, scale=scale)
                cv2.rectangle(frame, (0, rows - 80), (cols, rows), (0, 0, 0), -1)
                
             # It'll show how many people are there in that place   
            cv2.putText(frame,
                        "Total Persons : " + str(len(boxes)),
                        (25, 800),
                        fontFace=cv2.QT_FONT_NORMAL,
                        fontScale=1,
                        color=(0, 0, 255))
            
            # It'll show how many people violate the social distancing rule
            cv2.putText(frame,
                        "Defaulters : " + str(len(set(indexes))),
                        (425, 800),
                        fontFace=cv2.QT_FONT_NORMAL,
                        fontScale=1,
                        color=(0, 0, 255))
            
        # save the detection footage    
        if WRITE_VIDEO:
            out.write(frame)
        #  PRESS 'q' buttom to end the detection
        if SHOW_OUTPUT:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if WRITE_VIDEO:
        out.release()

    cv2.destroyAllWindows()
    cap.release()


# In[58]:


if __name__ == "__main__":
    # path to input video
    video = "pedestrians.mp4"
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        video_source = "pedestrians.mp4"

    main(video_source=video)


# In[ ]:





# In[ ]:




