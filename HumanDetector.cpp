#include "HumanDetector.h"

constexpr int FRAME_STEP = 8;

constexpr int HIGHT_INCREASE = 1.5;
constexpr int MIN_WIDTH_OBJECT = 15;
constexpr int MIN_HEIGHT_OBJECT = 30;

constexpr int MAX_WIDTH_OBJECT = 500;
constexpr int MAX_HEIGHT_OBJECT = 500;

HumanDetector::HumanDetector()
{
    m_fgbg = cv::createBackgroundSubtractorMOG2(50, 16, false);
    m_skipping_frame = 0;
}

void HumanDetector::detect(cv::Mat &frame)
{
    if (FRAME_STEP == m_skipping_frame++)
    {
        m_skipping_frame = 0;
        m_roi = calculate(frame);
        isAccum = true;
    }
    for (uint i = 0ul; i < m_roi.size(); ++i)
    {
        cv::rectangle(frame, m_roi[i], cv::Scalar(0, 0, 255), 5);
    }
}

std::vector<cv::Rect> HumanDetector::calculate(const cv::Mat& frame)
{
    cv::Mat fgmask;
    m_fgbg->apply(frame, fgmask);
    cv::Mat opening;
    cv::morphologyEx(fgmask,
                     opening,
                     cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    std::vector<std::vector<cv::Point>> rawContours;
    cv::findContours(opening, rawContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Rect> outRect;
    for (uint j = 0ul; j < rawContours.size(); ++j)
    {
        cv::Rect rect = cv::boundingRect(rawContours[j]);
        if (rect.height > rect.width * HIGHT_INCREASE   &&
            rect.width  >= MIN_WIDTH_OBJECT  && rect.width <= MAX_WIDTH_OBJECT &&
            rect.height >= MIN_HEIGHT_OBJECT && rect.height <= MAX_HEIGHT_OBJECT
           )
        {
            outRect.push_back(rect);
        }
    }
    return outRect;
}
