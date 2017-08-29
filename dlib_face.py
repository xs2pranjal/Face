import cv2
import dlib

vid = cv2.VideoCapture(0)

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while (True):
	ret, frame = vid.read()
	gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	
	detections = detect(gray,1)

	for k, d in enumerate(detections):
		shape = predict(gray, d)
		for i in range (1,68):
			cv2.circle(frame, (shape.part(i).x,shape.part(i).y),1,(0,255,0), thickness=2)
	cv2.imshow("image", cv2.flip(frame,1))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
