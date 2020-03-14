from collections import namedtuple
import numpy as np
from enum import Enum, unique





@unique
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1   = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba      = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15

    Generic_Enemy = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
class StaticTileType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top =  0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1 
    Coin = 0xC2
    Breakable_Block = 0x51

    Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

@unique
class DynamicTileType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift=  0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68

    Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

class ColorMap(Enum):
    Empty = (255, 255, 255)   # White
    Ground = (128, 43, 0)     # Brown
    Fake = (128, 43, 0)
    Mario = (0, 0, 255)
    Goomba = (255, 0, 20)
    Top_Pipe1 = (0, 15, 21)  # Dark Green
    Top_Pipe2 = (0, 15, 21)  # Dark Green
    Bottom_Pipe1 = (5, 179, 34)  # Light Green
    Bottom_Pipe2 = (5, 179, 34)  # Light Green
    Coin_Block1 = (219, 202, 18)  # Gold
    Coin_Block2 = (219, 202, 18)  # Gold
    Breakable_Block = (79, 70, 25)  # Brownish

    Generic_Enemy = (255, 0, 20)  # Red
    Generic_Static_Tile = (128, 43, 0) 
    Generic_Dynamic_Tile = (79, 70, 25)

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

        Player_Y_Pos_On_Screen = 0xCE
        Player_Vertical_Screen_Position = 0xB5

    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # We only care about enemies that are drawn. Others may?? exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        enemies = []

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num]
            # Is there an enemy? 1/0
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen #- ram[0x71c]
                # print(ram[0x71c])
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins)
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy-
                e = Enemy(0x6, location, tile_location)

                enemies.append(e)

        return enemies

    @classmethod
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)

    @classmethod
    def get_mario_score(cls, ram: np.ndarray) -> int:
        multipllier = 10
        score = 0
        for loc in range(0x07DC, 0x07D7-1, -1):
            score += ram[loc]*multipllier
            multipllier *= 10

        return score

    @classmethod
    def get_mario_location_on_screen(cls, ram: np.ndarray):
        mario_x = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Pos_On_Screen.value] * ram[cls.RAMLocations.Player_Vertical_Screen_Position.value] + cls.sprite.height
        return Point(mario_x, mario_y)

    @classmethod
    def get_tile_type(cls, ram:np.ndarray, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x
        y = mario.y + delta_y + cls.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc (status bar at top)
        if sub_page_y not in range(13):# or sub_page_x not in range(16):
            return StaticTileType.Empty.value

        addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x
        return ram[addr]

    @classmethod
    def get_tile_loc(cls, x, y):
        row = np.digitize(y, cls.ybins) - 2
        col = np.digitize(x, cls.xbins)
        return (row, col)

    @classmethod
    def get_tiles(cls, ram: np.ndarray):
        tiles = {}
        row = 0
        col = 0

        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x

        enemies = cls.get_enemy_locations(ram)
        y_start = 0
        mx, my = cls.get_mario_location_in_level(ram)
        my += 16
        # Set mx to be within the screen offset
        mx = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]

        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                tile = cls.get_tile(x_pos, y_pos, ram)
                x, y = x_pos, y_pos
                page = (x // 256) % 2
                sub_x = (x % 256) // 16
                sub_y = (y - 32) // 16                
                addr = 0x500 + page*208 + sub_y*16 + sub_x
                
                # PPU is there, so no tile is there
                if row < 2:
                    tiles[loc] =  StaticTileType.Empty
                else:

                    try:
                        tiles[loc] = StaticTileType(tile)
                    except:
                        tiles[loc] = StaticTileType.Fake
                    for enemy in enemies:
                        ex = enemy.location.x
                        ey = enemy.location.y + 8
                        # Since we can only discriminate within 8 pixels, if it falls within this bound, count it as there
                        if abs(x_pos - ex) <=8 and abs(y_pos - ey) <=8:
                            tiles[loc] = EnemyType.Generic_Enemy
                # Next col
                col += 1
            # Move to next row
            col = 0
            row += 1

        # Place marker for mario
        mario_row, mario_col = cls.get_mario_row_col(ram)
        loc = (mario_row, mario_col)
        tiles[loc] = DynamicTileType.Mario

        return tiles

    @classmethod
    def get_mario_row_col(cls, ram):
        x, y = cls.get_mario_location_on_screen(ram)
        # Adjust 16 for PPU
        y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        return (row, col)


    @classmethod
    def get_tile(cls, x, y, ram, group_non_zero_tiles=True):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16

        if sub_y not in range(13):
            return StaticTileType.Empty.value

        addr = 0x500 + page*208 + sub_y*16 + sub_x
        if group_non_zero_tiles:
            if ram[addr] != 0:
                return StaticTileType.Fake.value

        return ram[addr]

        
