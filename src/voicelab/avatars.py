"""Original procedural portraits: no external assets, accounts or downloads."""

import hashlib

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF


def avatar_icon(key: str, category: str, size: int = 112) -> QIcon:
    seed = hashlib.sha256(key.encode()).digest()
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 112, size / 112)
    background = QColor.fromHsv(seed[0] * 359 // 255, 75, 110)
    hair = QColor.fromHsv(seed[1] * 359 // 255, 70 + seed[2] % 80, 110 + seed[3] % 130)
    outfit = QColor.fromHsv(seed[4] * 359 // 255, 130, 135)
    skin = QColor(("#f4ceba", "#dba78b", "#a56e56", "#f5d8ca")[seed[5] % 4])
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(background)
    p.drawEllipse(QRectF(1, 1, 110, 110))
    clip = QPainterPath()
    clip.addEllipse(QRectF(2, 2, 108, 108))
    p.setClipPath(clip)
    long_hair = category in ("Женские", "Аниме", "Фэнтези", "Атмосфера")
    p.setBrush(hair.darker(115))
    p.drawRoundedRect(QRectF(23, 13, 66, 88 if long_hair else 67), 30, 30)
    p.setBrush(outfit)
    p.drawEllipse(QRectF(14, 82, 84, 66))
    p.setBrush(skin.darker(108))
    p.drawRoundedRect(QRectF(48, 72, 16, 23), 7, 7)
    p.setBrush(skin)
    p.drawEllipse(QRectF(25, 49, 13, 18))
    p.drawEllipse(QRectF(74, 49, 13, 18))
    p.drawRoundedRect(QRectF(31, 25, 50, 55), 23, 23)
    p.setBrush(hair)
    p.drawChord(QRectF(25, 12, 63, 53), 0, 180 * 16)
    fringe = QPolygonF([QPointF(29, 30), QPointF(82, 25), QPointF(77, 55),
                        QPointF(64, 38), QPointF(52, 55), QPointF(43, 37), QPointF(31, 55)])
    p.drawPolygon(fringe)
    eye = QColor.fromHsv(seed[6] * 359 // 255, 160, 190)
    p.setBrush(eye)
    for x in (44, 68):
        p.drawEllipse(QRectF(x - 3, 56, 6, 9))
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(x - 2, 57, 2, 3))
        p.setBrush(eye)
    p.setPen(QPen(QColor("#a56769"), 1.5))
    p.drawArc(QRectF(50, 66, 12, 7), 180 * 16, 180 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(227, 139, 150, 90))
    p.drawEllipse(QRectF(34, 65, 10, 5))
    p.drawEllipse(QRectF(68, 65, 10, 5))
    if category == "Кибер":
        p.setBrush(QColor("#20394b"))
        p.drawRoundedRect(QRectF(31, 52, 50, 14), 5, 5)
        p.setPen(QPen(QColor("#8dffdf"), 2))
        p.drawLine(37, 58, 75, 58)
    elif category == "Фэнтези":
        p.setBrush(outfit.lighter(170))
        p.drawPolygon(QPolygonF([QPointF(29, 33), QPointF(19, 12), QPointF(42, 24)]))
        p.drawPolygon(QPolygonF([QPointF(70, 24), QPointF(93, 12), QPointF(83, 33)]))
    elif category == "Игровые":
        p.setPen(QPen(outfit.lighter(170), 5))
        p.drawArc(QRectF(21, 15, 70, 78), 0, 180 * 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(outfit.lighter(160))
        p.drawRoundedRect(QRectF(20, 46, 10, 24), 4, 4)
        p.drawRoundedRect(QRectF(82, 46, 10, 24), 4, 4)
    p.end()
    return QIcon(canvas)
