import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PIL import Image
from PIL.ImageQt import ImageQt
from typing import Tuple
import sys
import numpy as np
from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config

tile_size = (16, 16)

def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.ram = None

    def get_tiles(self):
        return SMB.get_tiles_on_screen(self.ram)

    def _draw_region_of_interest(self, painter: QPainter) -> None:
        # Grab mario row/col in our tiles
        mario = SMB.get_mario_location_on_screen(self.ram)
        mario_row, mario_col = SMB.get_tile_loc(mario.x, mario.y)
        mario_row, mario_col = SMB.get_mario_row_col(self.ram)
        # Determine how many tiles up, down, left, right we need
        input_directions = (2, 2, 2, 2)  # @TODO: get configparser to parse this stuff
        up, down, left, right = input_directions
        min_row = max(0, mario_row - up)
        max_row = min(14, mario_row + down)
        min_col = max(0, mario_col - left)
        max_col = min(15, mario_col + right)
        
        x, y = min_col, min_row
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        tile_width, tile_height = (16, 16)  # @TODO: Get config parser to parse this

        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))
        painter.drawRect(x*tile_width + 5, y*tile_height + 5, width*tile_width, height*tile_height)



    def draw_tiles(self, painter: QPainter):
        # tiles = self.get_tiles()
        tiles = SMB.get_tiles(self.ram)
        enemies = SMB.get_enemy_locations(self.ram)
        # mario_row, mario_col = SMB.get_mario_row_col(self.ram)
        # tiles[(mario_row, mario_col)] = DynamicTileType(0xAA)
        # print(self.ram[0x500:0x69f+1])
        # print(SMB.get_tiles(self.ram))
        # SMB.get_tiles(self.ram)
        # print(SMB.get_tiles(self.ram))
        # mario = SMB.get_mario_location_in_level(self.ram)
        # x, y = mario.x, mario.y 
        # page = (x // 256) % 2
        # sub_page_x = (x % 256) // 16
        # sub_page_y = (y - 32) // 16 
        # mario_screen = SMB.get_mario_location_on_screen(self.ram)
        # tiles_left = mario_screen.x // 16
        # tiles_up = mario_screen.y // 16
        # print('page: {}, subx: {}, suby: {}'.format(page, sub_page_x, sub_page_y))
        # print('left: {}, above: {}'.format(tiles_left, tiles_up))
        # assert tiles.shape == (13,16)
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                width = tile_size[0]
                height = tile_size[1]
                x_start = 5 + (width * col)
                y_start = 5 + (height * row)

                loc = (row, col)
                tile = tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass

                painter.drawRect(x_start, y_start, width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_border(painter, self.size)
        if not self.ram is None:
            self.draw_tiles(painter)
            self._draw_region_of_interest(painter)

    def _update(self):
        self.update()
    
    

class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
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
            # print(pixmap.width(), pixmap.height())
            self.img_label.setPixmap(pixmap)
        # print('here')
            
        painter.end()

    def _update(self):
        self.update()



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.top = 150
        self.left = 150
        self.width = 1100
        self.height = 500

        self.title = 'Super Mario Bros AI'
        self.env = retro.make('SuperMarioBros-Nes', state='Level1-1')


        self.init_window()
        self.show()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self.keys = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
        self.i = 1
        self._timer.start(1000 // 60)

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (514, 480), self.config)
        self.game_window.setGeometry(QRect(1100-514, 0, 514, 480))
        self.game_window.setObjectName('game_window')
        # # Reset environment and pass the screen to the GameWindow
        screen = self.env.reset()
        self.game_window.screen = screen
        # self.game_window.update()

        self.viz_window = Visualizer(self.centralWidget, (1100-514, 480), self.config)
        self.viz_window.setGeometry(0, 0, 1100-514, 480)
        self.viz_window.setObjectName('viz_window')
        self.viz_window.ram = self.env.get_ram()

    def keyPressEvent(self, event):
        k = event.key()
        m = {
            Qt.Key_Right : 7,
            Qt.Key_C : 8,
            Qt.Key_X: 0,
            Qt.Key_Left: 6,
            Qt.Key_Down: 5
        }
        if k in m:
            self.keys[m[k]] = 1

    def keyReleaseEvent(self, event):
        k = event.key()
        m = {
            Qt.Key_Right : 7,
            Qt.Key_C : 8,
            Qt.Key_X: 0,
            Qt.Key_Left: 6,
            Qt.Key_Down: 5
        }
        if k in m:
            self.keys[m[k]] = 0

        


    def _update(self) -> None:
        self.i += 1
        # right =   np.array([0,0,0,0,0,0,0,1,0], np.int8)
        # nothing = np.array([0,0,0,0,0,0,0,0,0], np.int8)
        ret = self.env.step(self.keys)
        self.game_window.screen = ret[0]
        # self.viz_window.ram = self.env.get_ram()
        
        self.update()
        self.game_window._update()
        if self.i % 6 == 0:
            self.viz_window.ram = self.env.get_ram()
            self.viz_window._update()

        



if __name__ == "__main__":
    config = Config('settings.config')
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())