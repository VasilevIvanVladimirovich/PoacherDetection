#include "HumanDetector.h"

#define IS_COMBINE 0
constexpr int MIN_WIDTH_OBJECT = 15;
constexpr int MIN_HEIGHT_OBJECT = 30;
constexpr int MAX_WIDTH_OBJECT = 500;
constexpr int MAX_HEIGHT_OBJECT = 500;
constexpr int DISTANCE_CONTOUR = 300;
constexpr int DISTANCE_ALONG_X = 60;
constexpr int DISTANCE_ALONG_Y = 60;
constexpr int FRAME_STEP = 8;
constexpr int WRITE_ALL_VAL = 1;

HumanDetector::HumanDetector()
{
    m_fgbg = cv::createBackgroundSubtractorMOG2(50, 16, false);
    m_skipping_frame = 0;
    isAccum = false;
}

void HumanDetector::detect(cv::Mat &frame, cv::Mat &back)
{
    if (FRAME_STEP == m_skipping_frame++)
    {
        m_skipping_frame = 0;
        m_roi = calculate(frame);
        isAccum = true;
    }
    for (int i = 0ul; i < m_roi.size(); ++i)
    {
        cv::rectangle(frame, m_roi[i], cv::Scalar(0, 0, 255), 5);
    }

    if(isAccum)
    {
        m_back.copyTo(back);
    }
}

std::vector<cv::Rect> HumanDetector::calculate(const cv::Mat& frame)
{
    cv::Mat fgmask;
    m_fgbg->apply(frame, fgmask);
    cv::Mat opening, edges;
    morphologyEx(fgmask,
                 opening,
                 cv::MORPH_OPEN,
                 cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    cv::Canny(opening, edges, 175, 255);


//    edges.copyTo(m_back);
    m_back = cv::Mat::zeros(frame.rows,
                            frame.cols,
                            CV_8UC1);

    std::vector<std::vector<cv::Point> > rawContours;
    cv::findContours(edges, rawContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );

#if IS_COMBINE
    //! IF need combine vector

    //combining all contours into structure
    std::vector<Contour> vecContours;
    for(size_t i = 0; i < rawContours.size(); i++)
    {
        Contour contour;
        if(computeCentroid(rawContours[i], contour.centroid))
        {
            contour.contourPoints = rawContours[i];
            vecContours.push_back(contour);
        }
    }

    bool isMerged = true;
    int i = 0;
    //combine vectors if they are adjacent
    while(isMerged && i != vecContours.size())
    {
        isMerged = false;
        for (int j = 0ul; j < vecContours.size(); ++j)
        {
            if (i != j &&
                abs(sqrt
                    (vecContours[i].centroid.x - vecContours[j].centroid.x) *
                    (vecContours[i].centroid.x - vecContours[j].centroid.x) +
                    (vecContours[i].centroid.y - vecContours[j].centroid.y) *
                    (vecContours[i].centroid.y - vecContours[j].centroid.y)) <= DISTANCE_CONTOUR)
            {
                vecContours[i].contourPoints.insert( vecContours[i].contourPoints.end(),
                                                     vecContours[j].contourPoints.begin(),
                                                     vecContours[j].contourPoints.end());
                computeCentroid(vecContours[i].contourPoints, vecContours[i].centroid);
                vecContours.erase(vecContours.begin() + j);
                isMerged = true;
                i = 0;
                break;
            }
        }
        i++;
    }

    //! Output checked !!!
    {
        std::vector<std::vector<cv::Point>> contourDebug;

        for (int i = 0ul; i < vecContours.size(); ++i)
            contourDebug.push_back(vecContours[i].contourPoints);

        for (int i = 0ul; i < contourDebug.size(); ++i)
            cv::drawContours(m_back, contourDebug, i, cv::Scalar(255), 5);
    }
    //!

    std::vector<cv::Rect> outRect;
    //valid contours
    for (int j = 0ul; j < vecContours.size(); ++j)
    {
        cv::Rect rect = cv::boundingRect(vecContours[j].contourPoints);

        if(rect.height > rect.width * 1.5  &&
           rect.width >= MIN_WIDTH_OBJECT && rect.width <= MAX_WIDTH_OBJECT &&
           rect.height >= MIN_HEIGHT_OBJECT && rect.height <= MAX_HEIGHT_OBJECT
           )
        {
            outRect.push_back(rect);
        }
    }
    return outRect;
#endif

#if !IS_COMBINE
    //! Output checked !!!
    {
        for (int i = 0ul; i < rawContours.size(); ++i)
            cv::drawContours(m_back, rawContours, i, cv::Scalar(255), 5);
    }
    //!

    std::vector<cv::Rect> outRect;
    //valid contours
    for (int j = 0ul; j < rawContours.size(); ++j)
    {
        cv::Rect rect = cv::boundingRect(rawContours[j]);

        if(rect.height > rect.width * 1.5  &&
           rect.width >= MIN_WIDTH_OBJECT && rect.width <= MAX_WIDTH_OBJECT &&
           rect.height >= MIN_HEIGHT_OBJECT && rect.height <= MAX_HEIGHT_OBJECT
           )
        {
            outRect.push_back(rect);
        }
    }
    return outRect;
#endif
}
//https://stackoverflow.com/questions/9074202/opencv-2-centroid?rq=1
bool HumanDetector::computeCentroid(std::vector<cv::Point> contour,
                                    cv::Point& centroid)
{
    if (contour.size() > 2)
    {
        double doubleArea = 0;
        cv::Point p(0, 0);
        cv::Point p0 = contour.back();
        for (const cv::Point& p1 : contour)
        {
           double a = p0.cross(p1); //cross product, (signed) double area of triangle of vertices (origin,p0,p1)
           p += (p0 + p1) * a;
           doubleArea += a;
           p0 = p1;
        }

        if (doubleArea != 0)
        {
            centroid =  p * (1 / (3 * doubleArea) );
            return true;
        }
        return false;
    }
    return false;
}
