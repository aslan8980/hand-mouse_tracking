import cv2
import mediapipe as mp
import numpy as np
import time

from Quartz.CoreGraphics import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetIntegerValueField,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGMouseEventClickState,
    kCGHIDEventTap,
)

from Quartz import CGDisplayBounds, CGMainDisplayID


main_display = CGMainDisplayID()
screen_bounds = CGDisplayBounds(main_display)

screen_w = int(screen_bounds.size.width)
screen_h = int(screen_bounds.size.height)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


prev_x, prev_y = 0,0

FRAME_MARGIN = 120

PINCH_ON = 0.08
PINCH_OFF = 0.13

CLICK_TIME = 0.15
DRAG_TIME = 0.35


right_pinch_active = False
right_pinch_start = 0

drag_active = False
left_fist = False

point_start = 0
point_active = False


def move_mouse(x,y,drag=False):

    if drag:
        event = CGEventCreateMouseEvent(None,kCGEventLeftMouseDragged,(x,y),0)
    else:
        event = CGEventCreateMouseEvent(None,kCGEventMouseMoved,(x,y),0)

    CGEventPost(kCGHIDEventTap,event)


def mouse_click(x,y):

    down = CGEventCreateMouseEvent(None,kCGEventLeftMouseDown,(x,y),0)
    up = CGEventCreateMouseEvent(None,kCGEventLeftMouseUp,(x,y),0)

    CGEventPost(kCGHIDEventTap,down)
    CGEventPost(kCGHIDEventTap,up)


def mouse_double_click(x,y):

    down = CGEventCreateMouseEvent(None,kCGEventLeftMouseDown,(x,y),0)
    CGEventSetIntegerValueField(down, kCGMouseEventClickState, 2)
    CGEventPost(kCGHIDEventTap, down)

    up = CGEventCreateMouseEvent(None,kCGEventLeftMouseUp,(x,y),0)
    CGEventSetIntegerValueField(up, kCGMouseEventClickState, 2)
    CGEventPost(kCGHIDEventTap, up)


def is_fist(landmarks):

    fingers=[(8,6),(12,10),(16,14),(20,18)]
    folded=0

    for tip,pip in fingers:
        if landmarks[tip].y>landmarks[pip].y:
            folded+=1

    return folded>=3


def is_pointing(landmarks):

    index_up = landmarks[8].y < landmarks[6].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_down and ring_down and pinky_down


while True:

    success,frame=cap.read()
    if not success:
        break

    frame=cv2.flip(frame,1)

    h,w,_=frame.shape
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    cv2.rectangle(frame,(FRAME_MARGIN,FRAME_MARGIN),(w-FRAME_MARGIN,h-FRAME_MARGIN),(0,255,0),2)

    results=hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):

            label=results.multi_handedness[idx].classification[0].label
            landmarks=hand_landmarks.landmark

            if label=="Left":
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                left_fist = is_fist(landmarks)


        for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):

            label=results.multi_handedness[idx].classification[0].label

            if label!="Right":
                continue

            landmarks=hand_landmarks.landmark

            palm=landmarks[9]
            index=landmarks[8]
            thumb=landmarks[4]

            pinch_dist=np.sqrt(
                (index.x-thumb.x)**2+
                (index.y-thumb.y)**2+
                (index.z-thumb.z)**2
            )

            if not left_fist:

                cam_x=int(palm.x*w)
                cam_y=int(palm.y*h)

                screen_x=np.interp(cam_x,(FRAME_MARGIN,w-FRAME_MARGIN),(0,screen_w))
                screen_y=np.interp(cam_y,(FRAME_MARGIN,h-FRAME_MARGIN),(0,screen_h))

                screen_x=np.clip(screen_x,0,screen_w)
                screen_y=np.clip(screen_y,0,screen_h)

                curr_x=prev_x*0.7+screen_x*0.3
                curr_y=prev_y*0.7+screen_y*0.3

                move_mouse(curr_x,curr_y,drag_active)

                prev_x,prev_y=curr_x,curr_y


            if pinch_dist<PINCH_ON:

                if not right_pinch_active:
                    right_pinch_start=time.time()
                    right_pinch_active=True

            elif pinch_dist>PINCH_OFF:

                if right_pinch_active:

                    duration=time.time()-right_pinch_start

                    if duration<CLICK_TIME:
                        mouse_click(prev_x,prev_y)

                    elif duration>DRAG_TIME:

                        up=CGEventCreateMouseEvent(
                            None,
                            kCGEventLeftMouseUp,
                            (prev_x,prev_y),
                            0
                        )

                        CGEventPost(kCGHIDEventTap,up)
                        drag_active=False

                    right_pinch_active=False


            if right_pinch_active:

                hold=time.time()-right_pinch_start

                if hold>DRAG_TIME and not drag_active:

                    down=CGEventCreateMouseEvent(
                        None,
                        kCGEventLeftMouseDown,
                        (prev_x,prev_y),
                        0
                    )

                    CGEventPost(kCGHIDEventTap,down)
                    drag_active=True


            if is_pointing(landmarks):

                if not point_active:
                    point_start=time.time()
                    point_active=True

                if time.time()-point_start>0.4:
                    mouse_double_click(prev_x,prev_y)
                    point_active=False

            else:
                point_active=False


    cv2.imshow("Gesture Mouse",frame)

    if cv2.waitKey(1)&0xFF==27:
        break


cap.release()
cv2.destroyAllWindows()