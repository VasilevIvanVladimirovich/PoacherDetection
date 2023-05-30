#ifndef HUMANDETECTOR_H
#define HUMANDETECTOR_H

#include "Defs.h"

class HumanDetector
{
public:
    HumanDetector();
    ~HumanDetector() = default;

    struct Contour
    {
        std::vector<cv::Point> contourPoints;
        cv::Point centroid;
    };

    void detect(cv::Mat& frame, cv::Mat& back);
private:
        std::vector<cv::Rect> calculate(const cv::Mat& frame);

        Contour mergeContour(Contour c1, Contour c2);

        bool computeCentroid(std::vector<cv::Point> contour,
                             cv::Point& centroid);

private:
    cv::Ptr<cv::BackgroundSubtractor> m_fgbg;

    std::vector<cv::Rect> m_roi;

    int m_skipping_frame;
    bool isAccum;

    //! Debug
    cv::Mat m_back;
};

#endif // HUMANDETECTOR_H
