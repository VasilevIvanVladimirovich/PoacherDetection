#ifndef VIDEOTHREAD_H
#define VIDEOTHREAD_H

#include "Defs.h"
#include "HumanDetector.h"

class VideoThread : public QThread
{
    Q_OBJECT
public:
    VideoThread(QObject *parent = 0);

    void stop();
    void setFileName(std::string name);

private:
    QPixmap mat2pixmap(cv::Mat mat, bool isGray);

public slots:
    void run() override;

signals:
    void sendFrame(QPixmap frame, int i);

private:
    std::string m_fileName;
    bool m_running;
};

#endif // VIDEOTHREAD_H
