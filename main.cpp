#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cstring>

#include <fr/shunyaface.h>

#define DATABASE "db.txt"

using namespace cv;
using namespace std;


void usage () {
   fprintf(stderr, "usage: facedemo");
   fprintf(stderr, "     : Recognizes face stored in the database file.");
   fprintf(stderr, "usage: facedemo [name]");
   fprintf(stderr, "     : Stores face into the database");
   fprintf(stderr, "    name: name for storing into the face database.");
   fprintf(stderr, "\n");
}

int main(int argc, char *argv[])
{
    /* Variable to Store frame */
    Mat frame;
    VideoCapture cap;

    if (!cap.open(0))
        return 0;

    /* Print Usage once regardless of the correct or incorrect usage*/
    usage();
    
    for (;;) {
        cap >> frame;

        if ( frame.empty() ) break; // end of video stream

        if ( waitKey(10) == 27 ) break; // stop capturing by pressing ESC

        /* Initialize the variables */
        vector<FaceInfo> detFaces;
        vector<float> embeddings;
        Mat face;
        string name;

        if (! frame.data ) { // Check for invalid input
            cout <<"Could not open or find the image" << endl ;
            return -1;
        }

        /* Detect the face */
        detFaces = detectFace(frame);

        /* Check if faces are detected or not */
        if (detFaces.size() > 0) {
            cout<<"Face Detected!"<<endl;
            /* Align Face */
            face = alignFace(frame, detFaces[0]);
            /*Get embeddings from face*/
            embeddings = getEmbeddings(face);
            
            /* Store face in database */
            /* While storing the Face user is expected to enter name in command line arguments */
            if (argc > 1) {
                name = argv[1];
                int8_t ret = storeFace(embeddings,name,DATABASE);

                if (1 == ret) {
                    cout<<"Face is stored successfully"<<endl;
                    break;

                } else {
                    cout<<"There is some problem with storing face"<<endl;
                    break;
                }
            } 

            /* Find face from the database.*/
            name = findFace(embeddings, DATABASE );

            if (name.compare("NULL") != 0) {
                cout<<"Face Recognized: "<<name<<endl;

            } else {
                cout<<"No Face Recognized!"<<endl;
            }

        } else {
            cout<<"No Face Detected!!"<<endl;
        }
    }

    return 0;
}
