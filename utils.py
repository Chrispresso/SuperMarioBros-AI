from collections import namedtuple
import numpy as np
from enum import Enum, unique





@unique
class EnemyType(Enum):
    Green_Koopa = 0x00
    Red_Koopa   = 0x01
    Goomba      = 0x06

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
class StaticTileType(Enum):
    Empty = 0x00
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1  # @TODO I think one of the coin (?? block) is actually a mushroom
    Breakable_Block = 0x51

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
class DynamicTileType(Enum):
    Mario = 0xAA

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

class ColorMap(Enum):
    Empty = (255, 255, 255)   # White
    Ground = (128, 43, 0)     # Brown
    Mario = (0, 0, 255)
    Goomba = (255, 0, 20)
    Top_Pipe1 = (0, 15, 21)  # Dark Green
    Top_Pipe2 = (0, 15, 21)  # Dark Green
    Bottom_Pipe1 = (5, 179, 34)  # Light Green
    Bottom_Pipe2 = (5, 179, 34)  # Light Green
    Coin_Block1 = (219, 202, 18)  # Gold
    Coin_Block2 = (219, 202, 18)  # Gold
    Breakable_Block = (79, 70, 25)  # Brownish

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])

class Tile(object):
    __slots__ = ['type']
    def __init__(self, type: Enum):
        self.type = type

class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        enemy_type = EnemyType(enemy_id)
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location





class SMB(object):
    # SMB can only load 5 enemies to the screen at a time.
    # Because of that we only need to check 5 enemy locations
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = Shape(256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)
    resolution = Shape(256, 240)
    status_bar = Shape(width=resolution.width, height=2*sprite.height)

    xbins = list(range(16, resolution.width, 16))
    ybins = list(range(16, resolution.height, 16))


    @unique
    class RAMLocations(Enum):
        # Since the max number of enemies on the screen is 5, the addresses for enemies are
        # the starting address and span a total of 5 bytes. This means Enemy_Drawn + 0 is the
        # whether or not enemy 0 is drawn, Enemy_Drawn + 1 is enemy 1, etc. etc.
        Enemy_Drawn = 0x0F
        Enemy_Type = 0x16
        Enemy_X_Position_In_Level = 0x6E
        Enemy_X_Position_On_Screen = 0x87
        Enemy_Y_Position_On_Screen = 0xCF

        Player_X_Postion_In_Level       = 0x06D
        Player_X_Position_On_Screen     = 0x086

        Player_X_Position_Screen_Offset = 0x3AD
        Player_Y_Position_Screen_Offset = 0x3B8
        Enemy_X_Position_Screen_Offset = 0x3AE


    def read_byte(cls, ram: np.ndarray, location: int):
        pass

    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # We only care about enemies that are drawn. Others may?? exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        enemies = []

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num]
            # Is there an enemy 1/0?
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen - ram[0x71c]
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins) - 2
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy
                e = Enemy(enemy_id, location, tile_location)

                enemies.append(e)

        return enemies

    @classmethod
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[0xce] * ram[0xb5]
        return Point(mario_x, mario_y)

    @classmethod
    def get_mario_location_on_screen(cls, ram: np.ndarray):
        mario_x = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value] 
        mario_y = ram[0xce] * ram[0xb5] + cls.sprite.height  # @TODO: Change this to screen and not level
        return Point(mario_x, mario_y)

    @classmethod
    def get_tile_type(cls, ram:np.ndarray, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x  # @TODO maybe +1 or +4. Half seems to mess up hitboxes sometimes
        y = mario.y + delta_y + cls.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc (status bar at top)
        if sub_page_y not in range(13) or sub_page_x not in range(16):
            return 0x00
        addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x
        return ram[addr]

    @classmethod
    def get_tile_loc(cls, x, y):
        row = np.digitize(y, cls.ybins) - 2
        col = np.digitize(x, cls.xbins)
        return (row, col)

    # @classmethod
    # def get_tile_location(cls, ram: np.ndarray, )

    @classmethod
    def get_tiles_on_screen(cls, ram: np.ndarray):
        mario = cls.get_mario_location_in_level(ram)

        # How many tiles above and below mario are there?
        tiles_down = (cls.resolution.height - cls.status_bar.height - mario.y) // 16
        tiles_up = (13 - 1 - tiles_down)

        # How many tiles to the left and right of mario are there?
        x_loc = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]
        tiles_left = x_loc // 16
        tiles_right = 16 - 1 - tiles_left

        tiles_left = 7
        tiles_right = 8

        # tiles = np.empty((13,16), np.uint8)
        # tiles.fill(0xFF)
        tiles = {}

        # Grab enemies
        enemies = cls.get_enemy_locations(ram)

        for y in range(-tiles_up, tiles_down+1):
            for x in range(-tiles_left, tiles_right+1):
                dy = y*16
                dx = x*16
                
                loc = (y+tiles_up, x+tiles_left)
                if mario.x + dx - ram[0x71c] >= 256:
                    tiles[loc] = StaticTileType.Empty
                    continue

                tile_type = cls.get_tile_type(ram, dx, dy, mario)
                # @TODO fill in values
                if StaticTileType.has_value(tile_type):
                    tile = StaticTileType(tile_type)
                else:
                    # print('missing', tile_type)
                    tile = StaticTileType(0x00)
                tiles[loc] = tile
                # If dx and dy are both 0, this is where mario is
                # @TODO: I think this changes for when mario is big. He might take up 2 sprites then
                if dx == dy == 0:
                    tiles[loc]= DynamicTileType(0xAA)  # Mario

        for enemy in enemies:
            if enemy:
                ex = enemy.location.x
                if ex >= cls.resolution.width:
                    continue
                ey = enemy.location.y + 8
                ex += 8
                ybin = np.digitize(ey, cls.ybins) - 2
                xbin = np.digitize(ex, cls.xbins)
                loc = (ybin, xbin)
                tiles[loc] = EnemyType(enemy.type.value)


        return tiles