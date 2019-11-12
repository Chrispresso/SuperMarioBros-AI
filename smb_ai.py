import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PIL import Image
from PIL.ImageQt import ImageQt
from typing import Tuple
import sys

def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)

class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_border(painter, self.size)
        if not self.screen is None:
            original = Image.fromarray(self.screen)
            width = self.screen.shape[0] * 3 
            height = self.screen.shape[1] * 2
            resized = original.resize((width, height))
            # Create the image and label
            image = ImageQt(resized)
            qimage = QImage(image)
            # Center where the image will go
            x = (self.size[0] - width) // 2
            y = (self.size[1] - height) // 2
            # self.img_label.setGeometry(x, y, width, height)
            # Add image
            pixmap = QPixmap(qimage)
            print(pixmap.isNull())
            self.img_label.setPixmap(pixmap)
        
            



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.top = 150
        self.left = 150
        self.width = 700
        self.height = 500

        self.title = 'Super Mario Bros AI'
        self.env = retro.make('SuperMarioBros-Nes', state='Level1-1')


        self.init_window()
        self.show()

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (672, 480))
        self.game_window.setGeometry(QRect(0, 0, 672, 480))
        self.game_window.setObjectName('game_window')
        # Reset environment and pass the screen to the GameWindow
        screen = self.env.reset()
        self.game_window.screen = screen
        self.game_window.update()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())