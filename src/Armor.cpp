#include "Armor.h"

Armor::Armor() {}

void Armor::init(const cv::Mat& src)
{
    AREA_MAX = 200;
    AREA_MIN = 25;
    ERODE_KSIZE = 2;
    DILATE_KSIZE = 4;
    V_THRESHOLD = 230;
    S_THRESHOLD = 40;
    BORDER_THRESHOLD = 10;
    H_BLUE_LOW_THRESHOLD = 120;
    H_BLUE_LOW_THRESHOLD_MIN = 100;
    H_BLUE_LOW_THRESHOLD_MAX = 140;
    H_BLUE_HIGH_THRESHOLD = 180;
    H_BLUE_HIGH_THRESHOLD_MAX = 200;
    H_BLUE_HIGH_THRESHOLD_MIN = 160;
    H_BLUE_STEP = 1;
    H_BLUE_CHANGE_THRESHOLD_LOW = 5;
    H_BLUE_CHANGE_THRESHOLD_HIGH = 10;
    S_BLUE_THRESHOLD = 100;
    BLUE_PIXEL_RATIO_THRESHOLD = 40;//我感觉这个调太高还是不大好，最后这个还是配合深度图调
    getSrcSize(src);
    CIRCLE_ROI_WIDTH = srcW;
    CIRCLE_ROI_HEIGHT = srcH;
    CIRCLE_GRAY_THRESHOLD = 130;
    CIRCLE_AREA_THRESH_MAX = 10000;
    CIRCLE_AREA_THRESH_MIN = 30;
    target = cv::Point(320, 240);
    is_last_found = false;
    last_found_2 = false;
    last_found_3 = false;
    V_RATIO = 34;
    refresh_ctr = 0;
    V_element_erode = cv::getStructuringElement(
        cv::MORPH_CROSS, cv::Size(ERODE_KSIZE, ERODE_KSIZE));
    V_element_dilate = cv::getStructuringElement(
        cv::MORPH_CROSS, cv::Size(DILATE_KSIZE, DILATE_KSIZE));
    if (DRAW & SHOW_DRAW) {
        cout << "Show draw!" << endl;
        cv::namedWindow("draw", 1);
    }
    if (DRAW & SHOW_GRAY) {
        cout << "Show gray!" << endl;
        cv::namedWindow("gray", 1);
    }
    if (DRAW & SHOW_LIGHT_REGION) {
        cout << "Show light region" << endl;
        cv::namedWindow("light region", 1);
    }
}

void Armor::explore(const cv::Mat& src)
{
    double fps;
    static vector<cv::Mat> hsvSplit;
    static cv::Mat v_very_high;
    static vector<cv::RotatedRect> lights;
    static vector<vector<cv::Point> > V_contours;
    static vector<cv::Point2f> armors;
    fps = tic();
    if (DRAW & SHOW_DRAW)
        light_draw = src.clone();
    //if (is_last_found) {
    //++refresh_ctr;
    //findCircleAround(src);
    //cout << "FPS:" << 1 / (tic() - fps) << endl;
    //if (refresh_ctr == 500)
    //is_last_found = false;
    //return;
    //}
    //cleanAll();
    cvtHSV(src, hsvSplit);
    histV(hsvSplit[V_INDEX]);
    getLightRegion(hsvSplit, v_very_high);
    findContours(v_very_high, V_contours,
        CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    if (selectContours(V_contours, lights, hsvSplit) <= 0)
        cout << "No Lights Left" << endl;
    if (selectLights(lights, armors, src) <= 0)
        cout << "No armors Left" << endl;
    is_last_found = !armors.empty();
    cout << "FPS:" << 1 / (tic() - fps) << endl;
    fps = tic();

    lights.clear();
    armors.clear();
    V_contours.clear();
}

void Armor::track(const cv::Mat& src)
{
    double fps;
    fps = tic();
    last_found_3 = last_found_2;
    last_found_2 = is_last_found;
    if (DRAW & SHOW_DRAW)
        light_draw = src.clone();

    ++refresh_ctr;
    findCircleAround(src);
    cout << "FPS:" << 1 / (tic() - fps) << endl;
    if (refresh_ctr == 500) {
        is_last_found = false;
        refresh_ctr = 0;
    }
    return;
}
void Armor::getSrcSize(const cv::Mat& src)
{
    srcH = (int)src.size().height;
    srcW = (int)src.size().width;
}

void Armor::cvtHSV(const cv::Mat& src, vector<cv::Mat>& hsvSplit)
{
    static cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV_FULL);
    cv::split(hsv, hsvSplit);
}

void Armor::cvtGray(const cv::Mat& src, cv::Mat& gray)
{
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    cv::threshold(gray, gray, CIRCLE_GRAY_THRESHOLD,
        U8_MAX, cv::THRESH_BINARY);
    if (DRAW & SHOW_GRAY) {
        cv::imshow("gray", gray);
        cv::createTrackbar("THRESHOLD", "gray",
            &CIRCLE_GRAY_THRESHOLD, U8_MAX);
    }
}

void Armor::getLightRegion(vector<cv::Mat>& hsvSplit,
    cv::Mat& v_very_high)
{
    static cv::Mat s_low;
    cv::threshold(hsvSplit[V_INDEX], v_very_high,
        V_THRESHOLD, U8_MAX, cv::THRESH_BINARY);
    cv::threshold(hsvSplit[S_INDEX], s_low,
        S_THRESHOLD, U8_MAX, cv::THRESH_BINARY_INV);
    bitwise_and(s_low, v_very_high, v_very_high);
    cv::erode(v_very_high, v_very_high, V_element_erode);
    cv::dilate(v_very_high, v_very_high, V_element_dilate);
    if (DRAW & SHOW_LIGHT_REGION) {
        cout << "light region" << endl;
        cv::imshow("light region", v_very_high);
        //cv::imshow("S region", s_low);
        cv::createTrackbar("V_THRESHOLD", "light region",
            &V_THRESHOLD, U8_MAX);
        cv::createTrackbar("S_THRESHOLD", "light region",
            &S_THRESHOLD, U8_MAX);
    }
}

int Armor::selectContours(
    vector<vector<cv::Point> >& V_contours,
    vector<cv::RotatedRect>& lights,
    vector<cv::Mat>& hsvSplit)
{
    cv::RotatedRect rotated_rect;
    for (uint i = 0; i < V_contours.size(); i++) {
        if (isAreaTooBigOrSmall(V_contours[i]))
            continue;
        rotated_rect = minAreaRect(V_contours[i]);
        if (isCloseToBorder(rotated_rect))
            continue;
        if (!isBlueNearby(hsvSplit, V_contours[i]))
            continue;
        lights.push_back(rotated_rect);

        if (DRAW & SHOW_DRAW) {
            cv::Point2f vertices[4];
            rotated_rect.points(vertices);
            for (int j = 0; j < 4; ++j)
                line(light_draw, vertices[j], vertices[(j + 1) % 4],
                    cv::Scalar(0, 255, 0), 2);

            cv::imshow("draw", light_draw);
        }
    }
    return lights.size();
}

bool Armor::isAreaTooBigOrSmall(vector<cv::Point>& contour)
{
    int area = contourArea(contour);
    return (area > AREA_MAX || area < AREA_MIN);
}

bool Armor::isCloseToBorder(cv::RotatedRect& rotated_rect)
{
    double recx = rotated_rect.center.x;
    double recy = rotated_rect.center.y;
    return (recx < BORDER_THRESHOLD
        || recx > srcW - BORDER_THRESHOLD
        || recy < BORDER_THRESHOLD
        || recy > srcH - BORDER_THRESHOLD);
}

bool Armor::isBlueNearby(vector<cv::Mat>& hsvSplit,
    vector<cv::Point>& contour)
{
    int blue_pixel_cnt = 0;
    uchar pixel = 0;
    uchar pixel_min = H_BLUE_LOW_THRESHOLD_MAX;
    uchar pixel_max = H_BLUE_HIGH_THRESHOLD_MIN;
    uchar pixel_S_select = 0;
    for (int j = 0; j < (int)contour.size(); ++j) {
        pixel_S_select = *(hsvSplit[S_INDEX].ptr<uchar>(contour[j].y)
            + contour[j].x);
        pixel = *(hsvSplit[H_INDEX].ptr<uchar>(contour[j].y)
            + contour[j].x);
        if (pixel_S_select > S_BLUE_THRESHOLD
            && pixel < H_BLUE_HIGH_THRESHOLD
            && pixel > H_BLUE_LOW_THRESHOLD) {
            //cout << "pixel S:" << (int)pixel_S_select << " blue:" << (int)pixel << endl;
            if (pixel > pixel_max)
                pixel_max = pixel;
            if (pixel < pixel_min)
                pixel_min = pixel;
            ++blue_pixel_cnt;
        }
    }
    if (blue_pixel_cnt > 5) {
        pixel_max = pixel_max < H_BLUE_HIGH_THRESHOLD_MAX
            ? pixel_max
            : H_BLUE_HIGH_THRESHOLD_MAX;
        pixel_max = pixel_max > H_BLUE_HIGH_THRESHOLD_MIN
            ? pixel_max
            : H_BLUE_HIGH_THRESHOLD_MIN;
        pixel_min = pixel_min > H_BLUE_LOW_THRESHOLD_MIN
            ? pixel_min
            : H_BLUE_LOW_THRESHOLD_MIN;
        pixel_min = pixel_min < H_BLUE_LOW_THRESHOLD_MAX
            ? pixel_min
            : H_BLUE_LOW_THRESHOLD_MAX;
        if (pixel_min > H_BLUE_LOW_THRESHOLD + H_BLUE_CHANGE_THRESHOLD_HIGH)
            H_BLUE_LOW_THRESHOLD += H_BLUE_STEP;
        if (pixel_min < H_BLUE_LOW_THRESHOLD + H_BLUE_CHANGE_THRESHOLD_LOW)
            H_BLUE_LOW_THRESHOLD -= H_BLUE_STEP;
        if (pixel_max < H_BLUE_HIGH_THRESHOLD - H_BLUE_CHANGE_THRESHOLD_HIGH)
            H_BLUE_HIGH_THRESHOLD -= H_BLUE_STEP;
        if (pixel_max > H_BLUE_HIGH_THRESHOLD - H_BLUE_CHANGE_THRESHOLD_LOW)
            H_BLUE_HIGH_THRESHOLD += H_BLUE_STEP;
    }
    //cout << "blue cnt:" << blue_pixel_cnt << "/size" << contour.size() << endl;
    return float(blue_pixel_cnt) / contour.size() * 100
        > BLUE_PIXEL_RATIO_THRESHOLD;
}

int Armor::selectLights(
    const vector<cv::RotatedRect>& lights,
    vector<cv::Point2f>& armors,
    const cv::Mat& src)
{
    if (lights.size() <= 1)
        return -1;
    cv::Point2f pi;
    cv::Point2f pj;
    double midx;
    double midy;
    cv::Size2f sizei;
    //Size2f sizej = lights.at(j).size;
    double ai;
    //double b=sizei.height<sizei.width?sizei.height:sizei.width;
    double distance;
    static cv::Mat gray;
    cvtGray(src, gray);
    for (int i = 0; i < (int)lights.size() - 1; i++) {
        for (int j = i + 1; j < (int)lights.size(); j++) {
            pi = lights.at(i).center;
            pj = lights.at(j).center;
            midx = (pi.x + pj.x) / 2;
            midy = (pi.y + pj.y) / 2;
            sizei = lights.at(i).size;
            ai = sizei.height > sizei.width
                ? sizei.height
                : sizei.width;
            distance = sqrt((pi.x - pj.x) * (pi.x - pj.x)
                + (pi.y - pj.y) * (pi.y - pj.y));

            CIRCLE_ROI_WIDTH = int(distance * 1.5);
            CIRCLE_ROI_HEIGHT = int(ai * 3);
            //灯条距离合适
            if (distance < 1.5 * ai
                || distance > 7.5 * ai)
                continue;
            if (!isCircleInside(armors, gray, midx, midy))
                continue;
        }
    }

    chooseCloseTarget(armors);

    if (DRAW & SHOW_DRAW) {
        circle(light_draw, target, 3, cv::Scalar(0, 0, 255), -1);
        imshow("draw", light_draw);
        cv::createTrackbar("H_BLUE_LOW_THRESHOLD", "draw",
            &H_BLUE_LOW_THRESHOLD, U8_MAX);
        cv::createTrackbar("H_BLUE_HIGH_THRESHOLD", "draw",
            &H_BLUE_HIGH_THRESHOLD, U8_MAX);
        cv::createTrackbar("S_BLUE_THRESHOLD", "draw",
            &S_BLUE_THRESHOLD, U8_MAX);
        cv::createTrackbar("BLUE_PIXEL_RATIO_THRESHOLD", "draw",
            &BLUE_PIXEL_RATIO_THRESHOLD, 100);
    }
    return armors.size();
}

void Armor::chooseCloseTarget(vector<cv::Point2f>& armors)
{
    if (armors.empty())
        return;
    int closest_x = 0, closest_y = 0;
    int distance_armor_center = 0;
    int distance_last = sqrt(
        (closest_x - srcW / 2) * (closest_x - srcW / 2)
        + (closest_y - srcH / 2) * (closest_y - srcH / 2));
    for (uint i = 0; i < armors.size(); ++i) {
        //cout << "x:" << armors[i].x
        //<< "y:" << armors[i].y << endl;
        distance_armor_center = sqrt(
            (armors[i].x - srcW / 2) * (armors[i].x - srcW / 2)
            + (armors[i].y - srcH / 2) * (armors[i].y - srcH / 2));
        if (distance_armor_center < distance_last) {
            closest_x = (int)armors[i].x;
            closest_y = (int)armors[i].y;
            distance_last = distance_armor_center;
        }
    }
    target.x = closest_x;
    target.y = closest_y;
}

bool Armor::isCircleInside(vector<cv::Point2f>& armors,
    cv::Mat& gray, int midx, int midy)
{
    int rect_x = midx - CIRCLE_ROI_WIDTH / 2;
    rect_x = rect_x > 0 ? rect_x : 1;
    int rect_y = midy - CIRCLE_ROI_HEIGHT / 2;
    rect_y = rect_y > 0 ? rect_y : 1;
    int rect_w = midx + CIRCLE_ROI_WIDTH / 2;
    rect_w = rect_w > srcW
        ? (rect_w - srcW)
        : CIRCLE_ROI_WIDTH - 1;
    int rect_h = midy + CIRCLE_ROI_HEIGHT / 2;
    rect_h = rect_h > srcH
        ? (rect_h - srcH)
        : CIRCLE_ROI_HEIGHT - 1;
    cv::Mat roi_circle = gray(cv::Rect(rect_x, rect_y, rect_w, rect_h));
    vector<vector<cv::Point> > gray_contours;
    if (DRAW & SHOW_ROI)
        cv::imshow("roi", roi_circle);
    cv::findContours(roi_circle.clone(), gray_contours,
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < (int)gray_contours.size(); i++) {
        int area = contourArea(gray_contours[i]);
        if (area < 30)
            continue;
        cv::Point2f center(320, 240);
        float radius = 0;
        cv::minEnclosingCircle(gray_contours[i], center, radius);
        center.x += midx - CIRCLE_ROI_WIDTH / 2;
        center.y += midy - CIRCLE_ROI_HEIGHT / 2;
        float circleArea = PI * radius * radius;
        float r = area / circleArea;
        //cout << "Circle:" << r << endl;
        if (r > 0.7) {
            if (DRAW & SHOW_DRAW)
                cv::circle(light_draw, center, radius, cv::Scalar(0, 255, 255), 2);
            armors.push_back(center);
            return true;
        }
    }
    return false;
}

//void Armor::cleanAll()
//{
//V_contours.clear();
//armors.clear();
//}

bool Armor::isFound()
{
    return is_last_found||last_found_2||last_found_3;
}

int Armor::getTargetX()
{
    return target.x;
}

int Armor::getTargetY()
{
    return target.y;
}

void Armor::setDraw(int is_draw)
{
    DRAW = is_draw;
}

void Armor::findCircleAround(const cv::Mat& src)
{
    int rect_x = target.x - CIRCLE_ROI_WIDTH / 2;
    rect_x = rect_x > 0 ? rect_x : 1;
    int rect_y = target.y - CIRCLE_ROI_HEIGHT / 2;
    rect_y = rect_y > 0 ? rect_y : 1;
    int rect_w = target.x + CIRCLE_ROI_WIDTH / 2;
    rect_w = rect_w > srcW ? (rect_w - srcW) : CIRCLE_ROI_WIDTH - 1;
    int rect_h = target.y + CIRCLE_ROI_HEIGHT / 2;
    rect_h = rect_h > srcH ? (rect_h - srcH) : CIRCLE_ROI_HEIGHT - 1;
    cv::Mat roi_possible_circle = src(cv::Rect(rect_x, rect_y, rect_w, rect_h));

    //cv::cvtColor(roi_possible_circle, roi_possible_circle, CV_BGR2GRAY);
    cv::equalizeHist(roi_possible_circle, roi_possible_circle);
    cv::threshold(roi_possible_circle, roi_possible_circle,
        CIRCLE_GRAY_THRESHOLD, U8_MAX, cv::THRESH_BINARY);

    if (DRAW & SHOW_GRAY) {
        cv::imshow("gray", roi_possible_circle);
        cv::createTrackbar("THRESHOLD", "gray",
            &CIRCLE_GRAY_THRESHOLD, U8_MAX);
    }

    if (DRAW & SHOW_ROI)
        cv::imshow("roi", roi_possible_circle);

    vector<vector<cv::Point> > possible_circle_contours;
    cv::findContours(roi_possible_circle, possible_circle_contours,
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    cv::Point2f center(320, 240);
    for (uint i = 0; i < possible_circle_contours.size(); i++) {
        int area = cv::contourArea(possible_circle_contours[i]);
        //cout << "Area:" << area << endl;
        if (area > CIRCLE_AREA_THRESH_MAX)
            continue;
        if (area < CIRCLE_AREA_THRESH_MIN)
            continue;
        //if(area < 30)
        //continue;
        float radius = 0;
        cv::minEnclosingCircle(possible_circle_contours[i], center, radius);
        center.x += target.x - CIRCLE_ROI_WIDTH / 2;
        center.y += target.y - CIRCLE_ROI_HEIGHT / 2;
        float circleArea = PI * radius * radius;
        float r = area / circleArea;
        //cout << "Circle:" << r << endl;
        if (r > 0.7) {
            CIRCLE_AREA_THRESH_MAX = circleArea * 3;
            CIRCLE_AREA_THRESH_MIN = circleArea / 3;
            //cout << "Max:" << CIRCLE_AREA_THRESH_MAX;
            //cout << "Min:" << CIRCLE_AREA_THRESH_MIN;
            CIRCLE_ROI_WIDTH = radius * 8;
            CIRCLE_ROI_HEIGHT = radius * 4;
            target = center;
            is_last_found = true;
            if (DRAW & SHOW_DRAW) {
                //cv::circle(light_draw, center, radius, cv::Scalar(0, 255, 255), 2);
                //cv::circle(light_draw, target, 3, cv::Scalar(0, 0, 255), -1);
                cv::circle(light_draw, center, radius, cv::Scalar(255), 2);
                cv::circle(light_draw, target, 3, cv::Scalar(255), -1);
                imshow("draw", light_draw);
                cv::createTrackbar("H_BLUE_LOW_THRESHOLD", "draw",
                    &H_BLUE_LOW_THRESHOLD, U8_MAX);
                cv::createTrackbar("H_BLUE_HIGH_THRESHOLD", "draw",
                    &H_BLUE_HIGH_THRESHOLD, U8_MAX);
                cv::createTrackbar("S_BLUE_THRESHOLD", "draw",
                    &S_BLUE_THRESHOLD, U8_MAX);
                cv::createTrackbar("BLUE_PIXEL_RATIO_THRESHOLD", "draw",
                    &BLUE_PIXEL_RATIO_THRESHOLD, 100);
            }
            return;
        }
    }
    CIRCLE_ROI_WIDTH = srcW;
    CIRCLE_ROI_HEIGHT = srcH;
    CIRCLE_AREA_THRESH_MAX = 10000;
    CIRCLE_AREA_THRESH_MIN = 30;
    is_last_found = false;
    target = cv::Point2f(320, 240);
}

double Armor::tic()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((double)t.tv_sec + ((double)t.tv_usec) / 1000000.);
}

int Armor::histthre(cv::Mat& gray)
{
    cv::Mat hist;
	//Size core(3,3);  //转换为灰度图
	//GaussianBlur(gray,gray,core,0,0,BORDER_DEFAULT);
//	Mat grad_x, grad_y;
//	int scale = 1, delta = 0;
//	int ddepth = CV_16S;// 图像的深度
//	Sobel(gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//	Sobel(gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//	convertScaleAbs(grad_x, grad_x);
//    convertScaleAbs(grad_y, grad_y);
//    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gray);
//    namedWindow("Sobeled",1);
//    imshow("Sobeled", gray);

//    waitKey(0);
//    Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
//    erode(gray,gray, element);
//    Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//    filter2D(src, src, src.depth(), kernel);
//    cvtColor(src, gray, CV_BGR2GRAY);
//    namedWindow("gray",1);
//    imshow("gray",gray);
//    waitKey(0);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	int channels[] = {0};
	bool uniform = true;
	bool accumulate = false;

	/*计算直方图*/
	calcHist(&gray, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	/*创建描绘直方图的“图像”，和原图大小一样*/
	//int hist_w = gray.cols;
//	cout << "cols: " << hist_w << endl;
	//int hist_h = gray.rows;
//	cout << "rows: " << hist_h << endl;
	//一个点之间的距离，宽度是h，那么把宽度分成了256份

    cv::Mat histImage(100, 256*3, CV_8UC3, cv::Scalar(0, 0, 0));//设置为全黑的颜色

	/*直方图归一化范围，histImage.rows]*/
//	normalize(hist, hist, 100, 0, cv::NORM_MINMAX, -1, cv::Mat());

	/*画直线*/
//	for (int i = 1; i < histSize; ++i)
//	{
//		//cvRound：类型转换。 这里hist为256*1的一维矩阵，存储的是图像中各个灰度级的归一化值
//		line(histImage,
//             Point((i - 1)*3, 100 - cvRound(hist.at<float>(i - 1))),//（x,y）坐标
//			 Point((i)*3, 100 - cvRound(hist.at<float>(i))),
//			 Scalar(0, 0, 255), 2, 8, 0);
//	}
//    namedWindow("figure_src",1);
//	namedWindow("figure_hist",1);
//	imshow("figure_src", src);
//	imshow("figure_hist", histImage);
//	waitKey(0);
	int threshold_hist = 0;

	//寻找适当的二值化权值
	if (hist.isContinuous())
	{
		float * matpt = hist.ptr<float>(0);
//		cout << sum_hist << endl;
		float proper_hist = Size_of_Frame*0.618;
		float temp_hist = 0;

		for (int i = 255; i > -1; i--)
		{
			temp_hist += *(matpt+i);
			if (proper_hist <= temp_hist)
			{
				threshold_hist = i;
				break;
			}
		}

	}
//	Mat dst;
//	cout << threshold_hist << endl;
//	threshold(gray, dst, threshold_hist, 255, CV_THRESH_BINARY);
//	namedWindow("threshold",1);
//	imshow("threshold", dst);
//	imwrite("result_5.png",dst);
//	waitKey(0);
//	char delay;
//	cout << "Type in anything to quit" << endl;
//	cin >> delay;
//    cout << threshold_hist;
	return threshold_hist;
}

void Armor::histV(cv::Mat& V_channel)
{
	cv::Mat V_hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	int channels[] = {0};
	bool uniform = true;
	bool accumulate = false;

	/*计算直方图*/
	cv::calcHist(&V_channel, 1, channels, cv::Mat(), V_hist, 1, &histSize, &histRange, uniform, accumulate);
//    cv::calcHist(&hsvSplit[S_INDEX], 1, channels, cv::Mat(), S_hist, 1, &histSize, &histRange, uniform, accumulate);

	//一个点之间的距离，宽度是h，那么把宽度分成了256份

    cv::Mat V_histImage(100, 256, CV_8UC3, cv::Scalar(0, 0, 0));
	/*直方图归一化范围，histImage.rows]*/
//	cv::normalize(V_hist, V_hist, 100, 0, cv::NORM_MINMAX, -1, cv::Mat());
	/*画直线*/
//	for (int i = 1; i < histSize; ++i)
//	{
////		cvRound：类型转换。 这里hist为256*1的一维矩阵，存储的是图像中各个灰度级的归一化值
//        cv::line(V_histImage,cv::Point(i - 1, 100 - cvRound(V_hist.at<float>(i - 1))),//（x,y）坐标
//			 cv::Point(i, 100 - cvRound(V_hist.at<float>(i))),
//			 cv::Scalar(0, 0, 255), 2, 8, 0);
//	}
//    cv::line(V_histImage,cv::Point(V_THRESHOLD,0),cv::Point(V_THRESHOLD,100),cv::Scalar(255, 255, 255), 2, 8, 0);

    if (V_hist.isContinuous())
	{
		float * matpt = V_hist.ptr<float>(0);
		float sum_hist = 0;
//		for (int i = 0; i < 256; i++)
//		{
//			sum_hist += *(matpt+i);
////			cout<< *(matpt+i) << " this many pixels belong to the value: "<< i << endl;
//		}
//		cout << "The proper hist would be" << sum_hist << endl;
		float proper_hist = Size_of_Frame* V_RATIO;
		float temp_hist = 0;

		for (int i = 255; i > -1; i--)
		{
			temp_hist += *(matpt+i) * 1000;
			if (proper_hist <= temp_hist)
			{
				V_THRESHOLD = i;
				break;
			}
		}

	}

//	cv::namedWindow("V_Channel",1);
//	cv::namedWindow("V_hist",1);
//	cv::imshow("V_Channel", hsvSplit[V_INDEX]);
//	cv::imshow("V_hist", V_histImage);
//	cv::createTrackbar("V_RATIO", "V_Channel", &V_RATIO, 1000);
//	cv::waitKey(0);
}
