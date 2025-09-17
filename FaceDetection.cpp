#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <map>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Structure to store face information
struct FaceData {
    string name;
    steady_clock::time_point firstDetected;
    int detectionCount;
};

// Map to store known faces 
map<int, FaceData> knownFaces;
int nextFaceId = 0;

// Function to get a color based on face ID 
Scalar getColorFromId(int id) {
    vector<Scalar> colors = {
        Scalar(255, 0, 0),   // Blue
        Scalar(0, 255, 0),   // Green
        Scalar(0, 0, 255),   // Red
        Scalar(255, 255, 0), // Cyan
        Scalar(255, 0, 255), // Magenta
        Scalar(0, 255, 255), // Yellow
        Scalar(255, 255, 255) // White
    };
    return colors[id % colors.size()];
}

// Function to draw a rounded rectangle
void roundedRectangle(Mat& img, Point topLeft, Point bottomRight, const Scalar& color, int thickness, int lineType, int cornerRadius) {
    // Draw the main rectangles
    rectangle(img, Point(topLeft.x + cornerRadius, topLeft.y), Point(bottomRight.x - cornerRadius, bottomRight.y), color, thickness, lineType);
    rectangle(img, Point(topLeft.x, topLeft.y + cornerRadius), Point(bottomRight.x, bottomRight.y - cornerRadius), color, thickness, lineType);

    // Draw the corner circles
    circle(img, Point(topLeft.x + cornerRadius, topLeft.y + cornerRadius), cornerRadius, color, thickness, lineType);
    circle(img, Point(bottomRight.x - cornerRadius, topLeft.y + cornerRadius), cornerRadius, color, thickness, lineType);
    circle(img, Point(topLeft.x + cornerRadius, bottomRight.y - cornerRadius), cornerRadius, color, thickness, lineType);
    circle(img, Point(bottomRight.x - cornerRadius, bottomRight.y - cornerRadius), cornerRadius, color, thickness, lineType);
}

void main() {
    // Get user's name
    string userName;
    cout << "Enter your name: ";
    cin >> userName;

    VideoCapture video(0);
    CascadeClassifier facedetect;
    facedetect.load("haarcascade_frontalface_default.xml");

    Mat img;
    Mat overlay;

    
    namedWindow("Face Recognition System", WINDOW_NORMAL);

    while (true) {
        video.read(img);

        // Create a transparent overlay for UI elements
        overlay = Mat::zeros(img.size(), img.type());

        vector<Rect> faces;
        facedetect.detectMultiScale(img, faces, 1.3, 5);

        // Draw a header
        rectangle(overlay, Point(0, 0), Point(img.cols, 70), Scalar(50, 50, 255), FILLED);
        putText(overlay, "Face Recognition System", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
        putText(overlay, "Faces: " + to_string(faces.size()), Point(img.cols - 200, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);

        for (int i = 0; i < faces.size(); i++) {
            // Assign an ID to each face or use existing one
            int faceId = nextFaceId++;

            // Get a consistent color for this face
            Scalar faceColor = getColorFromId(faceId);

            // Draw rounded rectangle instead of regular rectangle
            roundedRectangle(img, faces[i].tl(), faces[i].br(), faceColor, 3, LINE_AA, 20);

            // Draw name tag background
            rectangle(overlay, Point(faces[i].x, faces[i].y - 40), Point(faces[i].x + 150, faces[i].y), faceColor, FILLED);

            // Display name 
            string displayName = (i == 0) ? userName : "Unknown";
            putText(overlay, displayName, Point(faces[i].x + 5, faces[i].y - 15), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 255, 255), 1);

            
            if (i > 0 && knownFaces.find(faceId) == knownFaces.end()) {
                cout << "New face detected! Please enter name for this person: ";
                string newName;
                cin >> newName;

                FaceData newFace;
                newFace.name = newName;
                newFace.firstDetected = steady_clock::now();
                newFace.detectionCount = 1;
                knownFaces[faceId] = newFace;
            }
        }

        double alpha = 0.7;
        addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

        imshow("Face Recognition System", img);

        // Check for ESC key to exit
        if (waitKey(1) == 27) {
            break;
        }
    }
}