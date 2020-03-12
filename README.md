
## Installation Instructions
You will need Python 3.6 or newer.

1. `cd /path/to/SuperMarioBros-AI`
2. Run `pip install -r requirements.txt`
3. Install the ROM
   - Go read the [disclaimer](https://wowroms.com/en/disclaimer)
   - Head on over to the ROM for [Super Mario Bros.](https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html) and click `download`.
4. Unzip `Super Mario Bros. (World).zip` to some location
5. Run `python -m retro.import "/path/to/unzipped/super mario bros. (world)"`
    - Make sure you run this on the folder, i.e. `python -m retro.import "c:\Users\chris\Downloads\Super Mario Bros. (World)"`
    - You should see output text:
      ```
      Importing SuperMarioBros-Nes
      Imported 1 games
      ```