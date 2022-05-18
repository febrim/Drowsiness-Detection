import cv2 #import library opencv
import dlib #import library dlib
from scipy.spatial import distance #import library distance dan mengambil scipy.partial

#menghitung jarak eclidean antara dua himpunan
def calculate_EAR(eye):

	#koordinat mata vertikal (x,y)
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])

	#koordinat jarak euclidan antara horizontal
	#koordinat mata (x,y)
	C = distance.euclidean(eye[0], eye[3])

	#menghitung rasio aspek mata
	ear_aspect_ratio = (A+B)/(2.0*C)

	#kemudian kita kembalikan aspek mata
	return ear_aspect_ratio

#menggunakan kamera
cap = cv2.VideoCapture(0)

#menggunakan face detector untuk mendetect muka
hog_face_detector = dlib.get_frontal_face_detector()

#memanggil file shape predictor untuk menggunakan facial landmark
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#kita looping frame dari video
while True:

	#read cameranya kemudian di resize, jika sudah di convert menjadi skala abu-abu (grayscale)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

	#kita looping pendeteksi muka (face detections)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
		
		#jika EAR kurang dari 0.26, menampilkan tulisan "DROWSY" dan "Are you Sleepy?"
        if EAR<0.26:
        	cv2.putText(frame,"DROWSY",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        	cv2.putText(frame,"Are you Sleepy?",(20,400),
        		cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        	print("Drowsy")
        print(EAR)

	#kita tampilkan framenya degan nama "Are you Sleepy?"
    cv2.imshow("Cek Mengantuk", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()