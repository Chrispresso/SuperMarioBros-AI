import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PIL import Image
from PIL.ImageQt import ImageQt
from typing import Tuple
import sys
import numpy as np
from utils import SMB

tile_size = (5, 5)

def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.ram = None

    def get_tiles(self):
        return SMB.get_tiles_on_screen(self.ram)

    def draw_tiles(self, painter: QPainter):
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_border(painter, self.size)
        self.draw_tiles(painter)
    

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
            # self.img_label = QtWidgets.QLabel(self.centralWidget)
            # screen = self.env.reset()
  
            width = self.screen.shape[0] * 3 
            height = int(self.screen.shape[1] * 2)
            # resized = original.resize((width, height))
            resized = self.screen
            original = QImage(self.screen, self.screen.shape[1], self.screen.shape[0], QImage.Format_RGB888)
            # Create the image and label
            # image = ImageQt(resized)
            qimage = QImage(original)
            # Center where the image will go
            x = (self.screen.shape[0] - width) // 2
            y = (self.screen.shape[1] - height) // 2
            self.img_label.setGeometry(0, 0, width, height)
            # Add image
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
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

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self.i = 1
        self._timer.start(1000 // 60)

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (672, 480))
        self.game_window.setGeometry(QRect(0, 0, 672, 480))
        self.game_window.setObjectName('game_window')
        # # Reset environment and pass the screen to the GameWindow
        screen = self.env.reset()
        print(SMB.get_tiles_on_screen(self.env.get_ram()))
        self.game_window.screen = screen
        # self.game_window.update()

    def _update(self) -> None:
        right = np.array([1,0,1,1,0,1,0,1,1], np.int8)
        nothing = np.array([0,0,0,0,0,0,0,0,0], np.int8)
        ret = self.env.step(nothing)
        self.game_window.screen = ret[0]

        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())