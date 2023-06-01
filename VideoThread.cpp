#include "VideoThread.h"

VideoThread::VideoThread(QObject *parent)
    : QThread(parent)
{
}

void VideoThread::stop()
{
    m_running = false;
}

void VideoThread::setFileName(std::string name)
{
    m_fileName = name;
}

QPixmap VideoThread::mat2pixmap(cv::Mat mat, bool isGray)
{
    if (!isGray)
    {
        return QPixmap::fromImage(
               QImage(mat.data,
                      mat.cols,
                      mat.rows,
                      mat.step,
                      QImage::Format_RGB888).rgbSwapped());
    }
    else
    {
        return QPixmap::fromImage(
               QImage(mat.data,
                      mat.cols,
                      mat.rows,
                      mat.step,
                      QImage::Format_Grayscale8));
    }

}

void VideoThread::run()
{
    m_running = true;
    cv::VideoCapture cap(m_fileName);
    if (!cap.isOpened())
    {
      qDebug()<<"Error opening video stream or file";
      m_running = false;
    }

    int fps = 1000 / 60;

    QElapsedTimer timer;
    timer.start();

    HumanDetector detector;

    while (m_running)
    {
        if (timer.elapsed() >= fps)
        {
            cv::Mat frame;
            // Capture frame-by-frame
            cap.read(frame);

            // If the frame is empty, break immediately
            if (frame.empty())
              break;

            detector.detect(frame);

            emit sendFrame(mat2pixmap(frame, false), 1);

            timer.start();
        }
    }
    cap.release();
}
