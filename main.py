import numpy as np
import cv2
import sys
import imutils
from pylsd import lsd
from random import randint
from segment import *
from copy import deepcopy
from getClass import *

process_stage = 0
prev_stage = -1
ix,iy = -1,-1
flagx = 0
edit_flag = 0
boxes=[]


def get_menubar():
    global process_stage;
    active_color  = (0,0,0)
    passive_color = (0,255,0)
    menubar = np.zeros((60,640,3),np.uint8)
    cv2.rectangle(menubar,(0,0),(640,60),(239,239,239),-1)
    cv2.putText(menubar, "Circuit Recognizer" ,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
    
    if process_stage == 0:
        #cv2.rectangle(menubar,(550,15),(620,45),(0,0,255),-1) ## next
        cv2.putText(menubar, "Segmentation" ,(230,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,active_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Classification" ,(330,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Result" ,(450,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.circle(menubar,(270,40),6,active_color,-1)
        cv2.circle(menubar,(370,40),6,passive_color,-1)
        cv2.circle(menubar,(470,40),6,passive_color,-1)
        #cv2.putText(menubar, "next" ,(570,35),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
    elif process_stage == 1:
        #cv2.rectangle(menubar,(550,15),(620,45),(0,0,255),-1) ## next
        cv2.putText(menubar, "Segmentation" ,(230,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Classification" ,(330,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,active_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Result" ,(450,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.circle(menubar,(270,40),6,passive_color,-1)
        cv2.circle(menubar,(370,40),6,active_color,-1)
        cv2.circle(menubar,(470,40),6,passive_color,-1)
        #cv2.putText(menubar, "next" ,(570,35),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
    else:
        #cv2.rectangle(menubar,(550,15),(620,45),(0,0,255),-1) ## next
        cv2.putText(menubar, "Segmentation" ,(230,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Classification" ,(330,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,passive_color,1,cv2.LINE_AA)
        cv2.putText(menubar, "Result" ,(450,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,active_color,1,cv2.LINE_AA)
        cv2.circle(menubar,(270,40),6,passive_color,-1)
        cv2.circle(menubar,(370,40),6,passive_color,-1)
        cv2.circle(menubar,(470,40),6,active_color,-1)
        #cv2.putText(menubar, "finish" ,(570,35),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
    return menubar

def draw_result_boxes(img,boxes):
    if process_stage == 0:
        for (x,y,w,h),_ in boxes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
    else:
        text = ['v_source','capacitor','ground','diode','resistor','inductor']
        font = cv2.FONT_HERSHEY_SIMPLEX
        for ((x,y,w,h),idx) in boxes:
            cv2.putText(img, text[idx] ,(x-5,y-5),font,0.6,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)

def get_v_s_orientation(x,y,w,h,pairs):
    lines = []
    for q in xrange(len(pairs)):
        midx1,midy1 = ((pairs[q,0]+pairs[q,2])/2,(pairs[q,1]+pairs[q,3])/2)
        if (midx1>x and midx1<x+w) and (midy1>y and midy1<y+h):
            if abs(pairs[q,0]-pairs[q,2]) > abs(pairs[q,1]-pairs[q,3]):
                angle_ax = 1    ## v_S vertical
                mid = midy1
            else:
                angle_ax = 0    ## v_s horizontal
                mid = midx1
            length = np.sqrt((pairs[q,0]-pairs[q,2])**2+(pairs[q,1]-pairs[q,3])**2)
            lines.append([mid,int(length)])
    lines = np.array(lines)
    lines = lines[lines[:,1].argsort()]
    if angle_ax == 0:
        if lines[1,0]>lines[0,0]:
            return 90
        else:
            return 270

    elif angle_ax ==1:
        if lines[1,0]>lines[0,0]:
            return 180
        else:
            return 0

def get_diode_orientation(x,y,w,h,pairs):
    lines = []
    angle_ax = 0
    for q in xrange(len(pairs)):
        midx1,midy1 = ((pairs[q,0]+pairs[q,2])/2,(pairs[q,1]+pairs[q,3])/2)
        if (midx1>x and midx1<x+w) and (midy1>y and midy1<y+h):
            if abs(pairs[q,0]-pairs[q,2]) > abs(pairs[q,1]-pairs[q,3]):
                angle_ax = 1    ## v_S vertical
                mid = midy1
            else:
                angle_ax = 0    ## v_s horizontal
                mid = midx1
            lines.append([mid])
    lines = np.array(lines)
    if angle_ax == 0:
        if abs(lines[0,0]-x) > abs(lines[0,0]-(x+w)):
            return 270
        else:
            return 90

    elif angle_ax ==1:
        if abs(lines[0,0]-y) > abs(lines[0,0]-(y+h)):
            return 0
        else:
            return 180

def output_file(wires,comp):
    counter  = np.zeros(6, dtype=np.int8)
    label    = ['voltage','cap','ground','diode','res','ind']
    abb      = ['V','C','G','D','R','L']
    offset   = [[0,16],[16,0],[0,0],[16,0],[16,16],[16,16]]
    filename = "{}.asc".format(str(sys.argv[1])[:-4])
    x = 0
    y = 0
    fo = open(filename,"w")
    fo.write("Version 4\n");
    fo.write("SHEET 1 880 680\n");
    for wire in wires:
        text = "WIRE {} {} {} {}\n".format(int(wire[0]),int(wire[1]),int(wire[2]),int(wire[3]))
        fo.write(text)

    for (box_id,type_id,n1_id,n2_id,(x1,y1),(x2,y2),angle) in comp:
        if type_id == 2: ## if comp is ground
            text = "FLAG {} {} 0\n".format(x1,y1)
            fo.write(text)
            continue

        if angle == 0:
            x = x1-offset[type_id][0]
            y = y1-offset[type_id][1]
        elif angle == 90:
            if (type_id == 0):
                x = x1+offset[type_id][1]*6
                y = y1
            elif (type_id == 3):
                x = x1+offset[type_id][0]*4
                y = y1-offset[type_id][0]
        elif angle == 270:
            if (type_id == 0):
                x = x1-offset[type_id][1]
                y = y1
            elif (type_id == 1) or (type_id == 3):
                x = x1
                y = y1+offset[type_id][0]
            elif (type_id == 4) or (type_id == 5):
                x = x1-offset[type_id][0]
                y = y1+offset[type_id][0]
        elif angle == 180:
            if(type_id == 0):
                x = x1+offset[type_id][0]
                y = y1+offset[type_id][1]*6
            elif(type_id==3):
                x = x1+offset[type_id][0]
                y = y1+64

        text1 = "SYMBOL {} {} {} R{}\n".format(label[type_id],x,y,angle)
        text2 = "SYMATTR InstName {}{}\n".format(abb[type_id],counter[type_id])
        fo.write(text1)
        fo.write(text2)
        counter[type_id] = counter[type_id]+1

    fo.close()

def svm_predict(th2,rects,boxes):
    
    
    for x,y,x2,y2 in rects:
        #region = cv2.resize(th2[y:y2,x:x2],(100,100),interpolation = cv2.INTER_CUBIC)
        if x<0:
            x=0
        if y<0:
            y=0

        if x2<0:
            x2=0

        if y2<0:
            y2=0

	try:
            prediction = predict(th2[y:y2,x:x2])
            print(prediction)
            if prediction:
                if prediction[0].startswith("0"):
                    idx=3
                elif prediction[0].startswith("1"):
                    idx=4
                else:
                    idx=5
            boxes.append([[int(x),int(y),int(x2-x),int(y2-y)],idx])
        except:
            pass
    return boxes

# mouse callback function
def mouse_event(event,x,y,flags,param):
    global process_stage,prev_stage,ix,iy,boxes,edit_flag
    boxes_t = []
    del_list = []
    edit = 0
    if event == cv2.EVENT_LBUTTONDBLCLK:
        prev_stage = process_stage
        process_stage = process_stage+1
    '''
    elif event == cv2.EVENT_RBUTTONDOWN:
        if process_stage == 0:
            ix,iy = x,y
        elif process_stage == 1:
            i = 0
            for (x0,y0,w,h),_ in boxes:
                if(x>x0 and y>y0 and x<x0+w and y<y0+h):
                    edit = i
                i = i+1

            cv2.namedWindow("edit")
            cv2.moveWindow("edit",960,100)
            cv2.setMouseCallback('edit',mouse_event_edit,edit)

            text = ['v_source','capacitor','ground','diode','resistor','inductor']
            edit = np.zeros((300,150,3),dtype=np.uint8)
            cv2.rectangle(edit,(0,0),(150,300),(255,255,255),-1)
            for i in xrange(len(text)):
                cv2.rectangle(edit,(20,i*40+20),(120,i*40+50),(0,0,255),-1)
                cv2.putText(edit, text[i] ,(30,i*40+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            while(edit_flag != 1):
                cv2.imshow("edit",edit)
                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            cv2.destroyWindow("edit")
            edit_flag = 0

    elif event == cv2.EVENT_RBUTTONUP:
        if process_stage == 0:
            bw = cv2.imread("data/skel.pgm",0)
            ends = skeleton_points(bw[iy:y,ix:x])
            for i in xrange(ends[0].size):
                ends[1][i] = ends[1][i]+ix
                ends[0][i] = ends[0][i]+iy
            v_pairs,h_pairs = lines_between_ends(ends)
            v_boxes = box_between_ends(v_pairs)
            h_boxes = box_between_ends(h_pairs)
            boxes_t = v_boxes + h_boxes

            if len(boxes_t) == 1:
                boxes.append(boxes_t[0])
            else:
                try:
                    th = cv2.imread("data/th.pgm",0)
                    rects = []
                    boxes_t = []
                    rects.append([ix,iy,x,y])
                    boxes_t = svm_predict(th,rects,boxes_t)
                    boxes.append(boxes_t[0])
                except:
                    pass

    elif event == cv2.EVENT_LBUTTONDOWN:
        if process_stage == 0:
            i = 0
            for (x0,y0,w,h),_ in boxes:
                if(x>x0 and y>y0 and x<x0+w and y<y0+h):
                    del_list.append(i)
                i = i+1
            del_list.sort(reverse=True)
            for i in del_list:
                del boxes[i]

'''

def mouse_event_edit(event,x,y,flags,param):
    global edit_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        text = ['v_source','capacitor','ground','diode','resistor','inductor']

        for i in xrange(len(text)):
            if(x>20 and y>i*40+20 and x<120 and y<i*40+50):
                boxes[param][1] = i
                edit_flag = 1
                break





if __name__ == "__main__":
    cv2.namedWindow("recognizer")
    cv2.moveWindow("recognizer",200,100)
    cv2.setMouseCallback('recognizer',mouse_event)

    src = cv2.imread(str(sys.argv[1]))
    src = imutils.resize(src,width=640)
    org = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ## endpoint operations
    try:
        img = cv2.GaussianBlur(gray,(9,9),0)
        th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
        th2 = th.copy()
        bw  = thinning(th)
        cv2.imwrite("data/skel.pgm",bw)
        cv2.imwrite("data/th.pgm",th2)
        ends = skeleton_points(bw)
    ## detection of ground, capacitor, v_source using lines; checking if parallel and ratio
    except:
        pass
       
    try:
        v_pairs,h_pairs = lines_between_ends(ends)
        v_boxes = box_between_ends(v_pairs)
        h_boxes = box_between_ends(h_pairs)
        boxes = v_boxes + h_boxes
    except:
        boxes=[]
    ## segmentation operations
    ## remove founded symbols and connection lines
    for ((x,y,w,h),idx) in boxes:
        th[y:y+h,x:x+w] = 0

    ## detect vert and hori lines then remove them from binary image
    lsd_lines = lsd(th)
    for line in lsd_lines:
        x1,y1,x2,y2,w = line
        angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
        if (angle<105 and angle>75) or angle>160 or angle<20:
            cv2.line(th,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),6)

    kernel = np.ones((11,11),np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    rects = []
    # Find Blobs on image  gives regions where componenets could be present
    cnts = cv2.findContours(closing.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for c in cnts:
        if cv2.contourArea(c)<80:    #threshold don't change unless required
        #if cv2.contourArea(c)<600:  #for larger images if components have large area 
            continue
        else:
            x,y,w,h = cv2.boundingRect(c)
            maxedge = max(w,h)
            x,y = int(((2*x+w)-maxedge)/2),int(((2*y+h)-maxedge)/2)
            rects.append([x-10,y-10,x+maxedge+10,y+maxedge+10])

    rects = non_max_suppression_fast(np.array(rects,dtype=float),0.1)

    boxes = svm_predict(gray,rects,boxes)

    while (1):
        src = org.copy()

        if process_stage == 2 and prev_stage == 1:
            ## find nodes and nodes end points
            for ((x,y,w,h),idx) in boxes:
                bw[y:y+h,x:x+w]  = 0
                th2[y:y+h,x:x+w] = 0

            node_closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
            node_temp = cv2.findContours(node_closing.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

            node_cnts = []
            node_ends = []
            pairs = np.vstack((v_pairs,h_pairs))
            i = 0
            for c in node_temp:
                if cv2.contourArea(c)>50:
                    color = (randint(0,255),randint(0,255),randint(0,255))

                    node_cnts.append((i,c))
                    ## draw each node contour and find ends point of node
                    node_mask = np.zeros(th.shape,np.uint8)
                    #cv2.drawContours(node_mask, [c] , 0, (255,255,255), 2)
                    cv2.drawContours(src,[c],0,(0,0,0),2)

                    node_thin = thinning(node_mask)
                    ends_n = skeleton_points(node_thin)
                    for j in xrange(ends_n[0].size):
                        
                        x,y = (ends_n[1][j],ends_n[0][j])
                        node_ends.append([i,[x,y]])
                        ##cv2.circle(src,(x,y),2,color,-1)

                    #cv2.circle(src,(20,20*i+20),7,color,-1)
                    #cv2.putText(src, str(i) ,(30,20*i+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,1,cv2.LINE_AA)
                    i = i+1

            

        draw_result_boxes(src,boxes)
        menubar = get_menubar()
        lastimg = np.vstack((src,menubar))
        cv2.imshow("recognizer", lastimg)
        #print(process_stage)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or flagx ==1 or process_stage==3:
            break

    while (process_stage <3):
        
        cv2.imshow("recognizer", lastimg)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
    

    cv2.destroyAllWindows()
