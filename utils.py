from collections import namedtuple
import numpy as np
from enum import Enum, unique




@unique
class Tile(Enum):
    EMPTY = 0

@unique
class Enemy(Enum):
    GREEN_KOOPA = 0x00
    RED_KOOPA   = 0x01
    GOOMBA      = 0x06

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])


class SMB(object):
    # SMB can only load 5 enemies to the screen at a time.
    # Because of that we only need to check 5 enemy locations
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = (256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)


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

    def read_byte(cls, ram: np.ndarray, location: int):
        pass

    def get_enemy_locations(cls, ram: np.ndarray):
        # We only care about enemies that are drawn. Others may?? exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn + enemy_num]
            # Is there an enemy 1/0?
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen

                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen + enemy_num]

    def get_tiles_on_screen(cls, ram: np.ndarray):
        # First we figure out where Mario is on the screen
        mario = cls.get_mario_location(ram)

    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen]
        mario_y = ram[0xce] * ram[0xb5] + cls.sprite.height
        return Point(mario_x, mario_y)

    def get_tile_type(cls, ram:np.ndarray, delta_x: int, delta_y: int):
        mario = cls.get_mario_location_in_level(ram)
        x = mario.x + delta_x + cls.sprite.width//2
        y = mario.y + delta_y - cls.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc
        addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x