#include "CustomGraphicsView.h"

CustomGraphicsView::CustomGraphicsView (QGraphicsScene* scene, QWidget* parent) :
    QGraphicsView(scene, parent) {}

CustomGraphicsView::CustomGraphicsView(QWidget* parent) :
    QGraphicsView(parent) {}


void CustomGraphicsView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        startX = event->x();
        startY = event->y();
    }
}

void CustomGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        QPointF oldp = mapToScene(startX, startY);
        QPointF newP = mapToScene(event->pos());
        QPointF translation = newP - oldp;

        translate(translation.x(), translation.y());
        startX = event->x();
        startY = event->y();
    }
}

void CustomGraphicsView::wheelEvent(QWheelEvent *event)
{
    if (event->angleDelta().y() > 0)
    {
        //event->position();
        QPoint p((int)event->pos().x(),(int)event->pos().y());
        QPointF oldp = mapToScene(p);
        scale(1.2,1.2);
        QPoint p2((int)event->pos().x(),(int)event->pos().y());
        QPointF newP = mapToScene(p2);
        QPointF translation = newP - oldp;
        translate(translation.x(), translation.y());
    }
    else
    {
        //event->position();
        QPoint p((int)event->pos().x(),(int)event->pos().y());
        QPointF oldp = mapToScene(p);
        scale(1/1.2,1/1.2);
        QPoint p2((int)event->pos().x(),(int)event->pos().y());
        QPointF newP = mapToScene(p2);
        QPointF translation = newP - oldp;
        translate(translation.x(), translation.y());
    }
}
