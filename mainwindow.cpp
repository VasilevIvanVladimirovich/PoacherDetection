#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
        auto mainLayout = new QHBoxLayout();

        //! Views
        m_video_scene = new QGraphicsScene();
        m_video_view = new CustomGraphicsView(m_video_scene);
        m_video_view->setMinimumSize(600, 400);
        m_video_view->verticalScrollBar()->blockSignals(true);
        m_video_view->horizontalScrollBar()->blockSignals(true);
        m_video_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_video_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_video_view->setTransformationAnchor(QGraphicsView::NoAnchor);

        m_sub_scene = new QGraphicsScene();
        m_sub_view = new CustomGraphicsView(m_sub_scene);
        m_sub_view->setMinimumSize(600, 400);
        m_sub_view->verticalScrollBar()->blockSignals(true);
        m_sub_view->horizontalScrollBar()->blockSignals(true);
        m_sub_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_sub_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_sub_view->setTransformationAnchor(QGraphicsView::NoAnchor);

        //! input_layout
        auto inputLayout = new QVBoxLayout();
        {
            auto selectedLayout = new QHBoxLayout();
            {
                m_inputPath_label = new QLabel();

                m_inputPath_label->setMinimumWidth(100);
                m_select_btn = new QPushButton();
                connect(m_select_btn,
                        &QPushButton::clicked,
                        this,
                        &MainWindow::onSelectedBtn_clicked);

                selectedLayout->addWidget(m_inputPath_label);
                selectedLayout->addWidget(m_select_btn);
            }

            m_start_btn = new QPushButton();
            m_start_btn->setCheckable(true);
            connect(m_start_btn,
                    &QPushButton::toggled,
                    this,
                    &MainWindow::onStartBtn_toggled);


            inputLayout->addLayout(selectedLayout);
            inputLayout->addWidget(m_inputPath_label);
            inputLayout->addWidget(m_start_btn);
        }

        mainLayout->addWidget(m_video_view);
        mainLayout->addWidget(m_sub_view);
        mainLayout->addLayout(inputLayout);

        QWidget* mainWidget = new QWidget;
        mainWidget->setLayout(mainLayout);

        retranslate();
        setCentralWidget(mainWidget);
}

MainWindow::~MainWindow()
{

}

void MainWindow::retranslate()
{
    m_inputPath_label->setText("Selected file");

    m_start_btn->setText("Start");
    m_select_btn->setText("...");
}

void MainWindow::updateView(QPixmap frame, int numberView)
{
    if (numberView == 1)
    {
        m_video_scene->clear();
        m_video_scene->addPixmap(frame);
        m_video_scene->update();
        m_video_view->setSceneRect(frame.rect());
    }
    if (numberView == 2)
    {
        m_sub_scene->clear();
        m_sub_scene->addPixmap(frame);
        m_sub_scene->update();
        m_sub_view->setSceneRect(frame.rect());
    }
}

void MainWindow::onSelectedBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home");


    m_fileName = fileName.toStdString();
    m_inputPath_label->setText(fileName);
}

void MainWindow::onStartBtn_toggled(bool toggled)
{
    //Start
    if (toggled)
    {
        m_videothread = new VideoThread(this);
        connect(m_videothread,
                SIGNAL(finished()),
                m_videothread,
                SLOT(deleteLater()));

        connect(m_videothread,
                &VideoThread::sendFrame,
                this,
                &MainWindow::updateView);


        m_videothread->setFileName(m_fileName);
        m_videothread->start();
    }
    //Finish
    else
    {

        m_videothread->stop();
    }
}

