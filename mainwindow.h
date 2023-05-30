#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "Defs.h"
#include "VideoThread.h"
#include "CustomGraphicsView.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void retranslate();

public slots:
    void updateView(QPixmap frame, int numberView);

private slots:
    void onSelectedBtn_clicked();
    void onStartBtn_toggled(bool toggled);

private:
    //!-> Variable
    std::string m_fileName;
    VideoThread *m_videothread;

    //!-> UI
    //! Video out
    QGraphicsScene *m_video_scene;
    CustomGraphicsView *m_video_view;

    //! Sub out widget
    QGraphicsScene *m_sub_scene;
    CustomGraphicsView *m_sub_view;

    QPushButton *m_start_btn;
    QPushButton *m_select_btn;

    QLabel *m_inputPath_label;
};
#endif // MAINWINDOW_H
