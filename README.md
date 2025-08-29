# Web-Based Face Recognition Attendance System

## Description

This project is a web application that automates the attendance process using real-time facial recognition. The system is built with a Python and Flask backend, uses OpenCV for computer vision, and employs a Scikit-learn model for face identification. It allows for easy registration of new users and logs attendance in a CSV file.



## Features

-   **Web-Based Interface:** Easy-to-use dashboard built with Flask.
-   **New User Registration:** Admins can add new users by uploading their photos.
-   **Real-Time Recognition:** Captures live video from a webcam to identify registered individuals.
-   **Automatic Attendance Logging:** Automatically records attendance with names and timestamps.
-   **Downloadable Reports:** Allows users to download the attendance sheet as a CSV file.

## Tech Stack & Requirements

-   **Language:** Python 3.x
-   **Web Framework:** Flask
-   **Computer Vision:** OpenCV
-   **Machine Learning:** Scikit-learn (`KNeighborsClassifier`)
-   **Data Handling:** Pandas, NumPy
-   **Model Persistence:** Joblib

To install the necessary libraries, run the following command:
```bash
pip install -r requirements.txt
```
*(Note: You should create a `requirements.txt` file listing all the libraries)*

## Setup and Installation

Follow these steps to set up the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    cd your-project-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install opencv-python flask scikit-learn numpy pandas joblib
    ```

## Usage

1.  **Train the Model:**
    *(Explain how to add new users and train the face recognition model. For example, you might need to place user images in a specific folder.)*
    Example: Place images of each person in the `static/faces/` directory.

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  **Open your browser** and navigate to `http://127.0.0.1:5000`.

4.  Click on "Take Attendance" to start the camera and begin the recognition process. The attendance will be logged in an `attendance.csv` file.
