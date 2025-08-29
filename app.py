import cv2
import os
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Constants
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Haarcascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories
for folder in ['Attendance', 'static', 'static/faces']:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# Create attendance CSV if not exists
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')


# -------- Utility Functions -------- #

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return faces
    except Exception as e:
        print(f"Face Extraction Error: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if faces:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        print("Model Trained Successfully!")
    else:
        print("No Faces Found for Training!")

def extract_attendance():
    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    return names, rolls, times, len(df)

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

def deletefolder(duser):
    pics = os.listdir(duser)
    for pic in pics:
        os.remove(duser + '/' + pic)
    os.rmdir(duser)

# -------- Routes -------- #

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    current_time = datetime.now().strftime("%H:%M:%S")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2, current_time=current_time)

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, 
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)
    
    if os.listdir('static/faces') == [] and os.path.exists('static/face_recognition_model.pkl'):
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except Exception as e:
        print(f"Training Error: {e}")

    flash('User deleted successfully!', 'info')
    return redirect(url_for('listusers'))

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash('No trained model found. Please add a user first.', 'danger')
        return redirect(url_for('home'))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash('Webcam not accessible!', 'danger')
        return redirect(url_for('home'))

    recognized_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            flash('Failed to capture frame from webcam.', 'danger')
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            try:
                identified_person = identify_face(face.reshape(1, -1))[0]
                if identified_person not in recognized_faces:
                    add_attendance(identified_person)
                    recognized_faces.append(identified_person)
                cv2.putText(frame, identified_person, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception as e:
                print(f"Recognition Error: {e}")

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) == 27:  # ESC Key
            break

    cap.release()
    cv2.destroyAllWindows()

    flash('Attendance marking completed.', 'success')
    return redirect(url_for('home'))

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if j % 5 == 0:
                imgname = f"{newusername}_{i}.jpg"
                cv2.imwrite(os.path.join(userimagefolder, imgname), frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    flash('User added and model trained successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/download')
def download():
    return send_file(attendance_file, as_attachment=True)

# -------- Main -------- #

if __name__ == '__main__':
    app.run(debug=True)
