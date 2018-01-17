#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Armor.h"
#include "Serial.h"
#include "GlobalCamera.h"

#include <fstream>


int main(int argc, char* argv[])
{

//#if PLATFORM == MANIFOLD
    //cv::VideoCapture cap(0);
//#else
    //cv::VideoCapture cap("/home/jachinshen/视频/Robo/train.avi");
//#endif
    GlobalCamera cap;
    cap.init();
    //if (!cap.isOpened()) {
        //ROS_INFO("camera not open");
        //return -1;
    //}
    cv::Mat src;
    cap.read(src);
    //cv::namedWindow("frame", 1);
    //cv::imshow("frame", src);
    //cv::waitKey(0);
    Armor armor;
    armor.setDraw(SHOW_ALL);
    armor.init(src);
    Serial serial;
    serial.init();
    while (1) {
        //cap >> src;
        cap.read(src);
        if (src.empty())
            break;

        cv::imshow("frame", src);
        //if (armor.isFound()) {
            armor.track(src);
        //} else {
        //    armor.explore(src);
        //}
        cout << armor.isFound()
             << " " << armor.getTargetX()
             << " " << armor.getTargetY() << endl;

        serial.sendTarget(armor.getTargetX(), armor.getTargetY(), armor.isFound());

        cv::waitKey(1);
    }
    //cap.release();
}
