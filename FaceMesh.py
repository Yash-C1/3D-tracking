import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    results = faceMesh.process(img)

    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img,face_landmark, mp_face_mesh.FACEMESH_CONTOURS, mp_draw.DrawingSpec((0,255,0),1,1), mp_draw.DrawingSpec((0,0,255),1,1))

    print(results.multi_face_landmarks)
    cv2.imshow("Face mesh",img)
    cv2.waitKey(1)
