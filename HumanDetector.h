#ifndef HUMANDETECTOR_H
#define HUMANDETECTOR_H

//! OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

class HumanDetector
{
public:
    HumanDetector();
    ~HumanDetector() = default;

    void detect(cv::Mat& frame);

private:
    std::vector<cv::Rect> calculate(const cv::Mat& frame);

private:
    cv::Ptr<cv::BackgroundSubtractor> m_fgbg;

    std::vector<cv::Rect> m_roi;

    uint8_t m_skipping_frame;
    bool isAccum;
};

#endif // HUMANDETECTOR_H
