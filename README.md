If you want to see a YouTube video describing this at a high level, and showcasing what was learned, take a look [here](https://www.youtube.com/watch?v=CI3FRsSAa_U).<br>
If you want to see my blog explaining how all of this works in great detail, go [here](https://chrispresso.github.io/AI_Learns_To_Play_SMB_Using_GA_And_NN).

<b>Update</b>:
The AI has successfully completed 1-1, 2-1, 3-1, 4-1, 5-1, 6-1, and 7-1.
It was also able to learn: flagpole glitch with an enemy, walljump, and a fast acceleration.

This contains information on the following:
- [Installation Instructions](#installation-instructions)
- [Command Line Options](#command-line-options)
  - [Config](#config)
  - [Loading Individuals](#loading-individuals)
  - [Replaying Individuals](#replaying-individuals)
  - [Disable Displaying](#disable-displaying)
  - [Debug](#debug)
- [Running Examples](#running-examples)
- [Creating a New Population](#creating-a-new-population)
- [Understanding the Config File](#understanding-the-config-file)
  - [Neural Network](#neural-network)
  - [Graphics](#graphics)
  - [Statistics](#statistics)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Mutation](#mutation)
  - [Crossover](#crossover)
  - [Selection](#selection)
  - [Misc](#misc)
- [Viewing Statistics](#viewing-statistics)
- [Results](#results)

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
## Command Line Options
If you want to see all command line options, the easiest way is to run `python smb_ai.py -h`.

### Config
- `'-c', '--config' FILE`. This sets the config file for all AI in that population. Normally if you load/replay an individual, the config file is taken from whatever folder you are loading/replaying. If you want to specifically set it, you can do so. This is also necessary to specify when creating a new population.

### Loading Individuals
You may want to load individuals. This could be due to your computer crashing and you wanting to load part of an old population. You may want to run experiments by combining individuals from different populations. Whatever the case, you can do so by specify *both* of the below arguments:
- `--load-file FOLDER`. Indicate the `/path/to/population/folder` that you want to load from.
- `--load-inds INDS`. This dictates which individuals from the folder to actually load. If you want a range, you can do `[2,100]`, where it will load individuals `best_ind_gen2, best_ind_gen3, ..., best_ind_gen100`. If you want to specify certain ones you can do `5,10,15,12,901` where they are seperated by a comma and no space.

Note that loading individuals only supports loading the best performing one from generations.

### Replaying Individuals
This is helpful if you want to watch particular individuals replay their run. You must specify *both* of the below arguments:
- `--replay-file FOLDER`. Indicate the `/path/to/population/folder` that you want to replay from.
- `--replay-inds INDS`. This accepts the same syntax as `--load-inds` with one additional syntax. If you want to replay starting at a certain individual through the end of the folder, you can do `[12,]` and this will begin at `best_ind_gen12` and run through the entire folder. *NOTE*: this is not supported in `--load-inds` since when you load it will be treated as an initially population and those individuals will be treated as parents. Because of that, you may accidentally have many more parents in your population gene pool than intended. 

### Disable Displaying
You are unfortunately limited by the refresh rate of your monitor for certain things in `PyQt`. Because of this, when the display is open (whether it's hidden or not) you can only run at the refresh rate of your monitor. The emulator supports faster updates and because of that an option has been created to run this through only command line. This can help speed up training.
- `--no-display`. When this option is present, nothing will be drawn to the screen.

### Debug
If you wish to know when populations are improving or when individuals have won, you can set a debug flag. This is helpul if you have disabled the display but wish to know how your population is doing.
- `--debug`. When this option is present, certain information regarding generation, individuals who have won, new max distance, etc. will print to the command line.

## Running Examples
I have several folders for examples. If you want to run any them to see how they perform, do:
  - `python smb_ai.py --replay-file "Example world1-1" --replay-inds 1213,1214`. This will load `settings.config` from the `Example world1-1` folder and replay the best individual from generation 1213 and 1214. 
  - `python smb_ai.py --replay-file "Example world4-1" --replay-inds 2259`. This will load `settings.config` from the `Example world4-1` folder and replay the best individual from generation 2259.

## Creating a New Population
If you want to create a new population, it's pretty easy. Make sure that if you are using the default `settings.config` that you change `save_best_individual_from_generation` and `save_population_stats` to reflect where you want that information saved. Once you have the config file how you want, simply run `python smb_ai.py -c settings.config` with any additional command line options.

## Understanding the Config File
The config file is what controls the initialization, graphics, save locations, genetic algorithm parameters and more. It's important to know what the options are and what you can change them to.

### Neural Network
Specified by `[NeuralNetwork]`
- `input_dims :Tuple[int, int, int]`. This defines the "pink box". The parameters are `(start_row, width, height)` where `start_row` begins at `0` at the top of the screen and increases going toward the bottom of the screen. `width` defines how wide the box is, beginning at `start_row`. `height` defines how tall the box is. Currently I do not support a `start_col`. This is set to be wherever Mario is at.
- `hidden_layer_architecture :Tuple[int, ...]`. Describes how many hidden nodes are in each hidden layer. `(12, 9)` would create two hidden layers. The first with `12` nodes and the second with `9`. This can be any length of 1 or more.
- `hidden_node_activation :str`. Options are `(relu, sigmoid, linear, leaky_relu, tanh)`. Defines what activation to use on hidden layers.
- `hidden_node_activation :str`. Options are `(relu, sigmoid, linear, leaky_relu, tanh)`. Defines what activation to use on hidden layers.
- `encode_row :bool`. Whether or not to have one-hot encoding to describe Mario's row location.

### Graphics
Specified by `[Graphics]`.
- `tile_size :Tuple[int, int]`. The size in pixels in (X, Y) direction to draw the tiles on the screen.
- `neuron_radius :float`. Radius to draw nodes on the screen.

### Statistics
Specified by `[Statistics]`.
- `save_best_individual_from_generation :str`. A folder location `/path/to/save/generation` to save best individuals.
- `save_population_stats :str`. `/file/location/of/stats.csv` where you wish to save statistics.

### Genetic Algorithm
Specified by `[GeneticAlgorithm]`.
- `fitness_func :lambda`. This is a function which will receive:
~~~python
def fitness_func(frames, distance, game_score, did_win):
  """
frames :int : Number of frames that Mario has been alive for
distance :int : Total horizontal distance gone through the level
game_score :int : Actual score Mario has received in the level through power-ups, coins, etc.
did_win :bool : True/False if Mario beat the level
"""
~~~
Because it is passed as a `lambda function`, there is only a return. This means no `if-statements`. This is why I use things like `max` and `min`. Whatever you choose, it is best to have something like `max(<your logic>, 0.00001)`. This will prevent certain problems if you choose `roulette selection` involving negative numbers.

### Mutation
Specified by `[Mutation]`.
- `mutation_rate :float`. Value must be between `[0.00, 1.00)`. Specifies the probability that *each* gene will mutate. In this case every trainable parameter is a gene.
- `mutation_rate_type :str`. Options are `(static, dynamic)`. `static` mutation will always be the same, while `dynamic` will decrease as the number of generations increase
- `gaussian_mutation_scale :float`. When a mutation occurs it is a normal gaussian mutation `N(0, 1)`. Because the parameters are capped between `[0.00, 1.00)`, a scale is provided to narrow this. The mutation would then be `N(0, 1) * scale`.

### Crossover
Specified by `[Crossover]`.
- `probability_sbx :float`. This is the probability to perform Simulated Binary Crossover (SBX). This is always `1.0` as of now because I do not support other types for this problem.
- `sbx_eta :int`. A bit of a complicated parameter, but the smaller the value, the more variance when creating offspring. As the parameter increases, the variance decreases and offspring are more centered around parent values. `100` still has variance but centers more around the parents. This helps a gene be able to change gradually rather than abruptly.
- `crossover_selection :str`. Options are `(roulette, tournament)`. `roulette` sums all individual fitness and gives each individual a probability to be selected for reproduction based on their fitness divided by total fitness of the population. `tournament` selection will randomly pick `n` individuals from the population and then select the one with the highest fitness from that subset.
- `tournament_size :int`. If you are using `crossover_selection = tournament`, then this value is used, otherwise it is ignored. Controls the number of individuals to pick from the population to form a subset to be selected from.

### Selection
Specified by `[Selection]`.
- `num_parents :int`. The number of individuals to begin with.
- `num_offspring :int`. The number of offspring that will be produced at the end of each generation.
- `selection_type :str`. Options are `(comma, plus)`. Let's say we define `<num_parents> <selection_type> <num_offspring>` and compare `(50, 100)` and `(50 + 100)`:
  - `(50, 100)` will begin generation 0 with 50 parents and then at the end of each generation produce 100 offspring from the parents. At generation 1, then, you will have 100 parents. No best individuals get carried over since all individuals for the next generation are simply offspring.
  - `(50 + 100)` will begin generation 0 with 50 parents and then at the end of each generation carry over the best 50 individuals from that generation *and* produce 100 offspring. At generation 1, then, you will have 150 parents. In this case 50 individuals get carried over to the next generation.
- `lifespan :float`. Really an int but considered a float to allow for `inf`. This dictates how long a certain individual is allowed to be in the population before dying off. In ths case of `selection_type = plus`, this would mean that an individual can only reproduce for a given number of generations before it dies off. For `selection_type = comma`, this value doesn't matter as no best performing individuals get carried over to the next generation.

### Misc
Specified by `[Misc]`.
- `level :str`. The current options are `(1-1, 2-1, 3-1, 4-1, 5-1, 6-1, 7-1, 8-1)` More can be supported by adding `state` information for the `gym environment`.
- `allow_additional_time_for_flagpole :bool`. Generally as soon as Mario touches the flag, he dies. This is just because he wins and there's no point in continuing the animation from there. You may wish to allow some additional time just to see it happen. I use this so I can record him completing the level.

## Viewing Statistics
The .csv file contains information on the `mean, median, std, min, max` for `frames, distance, fitness, wins`. If you want to view the max distance for a .csv you could do:
~~~python
stats = load_stats('/path/to/stats.csv')
stats['distance']['max']
~~~

Here is an example on how to plot stats using `matplotlib.pyplot`:
~~~python
from mario import load_stats
import matplotlib.pyplot as plt

stats = load_stats('/path/to/stats.csv')
tracker = 'distance'
stat_type = 'max'
values = stats[tracker][stat_type]

plt.plot(range(len(values)), values)
ylabel = f'{stat_type.capitalize()} {tracker.capitalize()}' 
plt.title(f'{ylabel} vs. Generation')
plt.ylabel(ylabel)
plt.xlabel('Generation')
plt.show()
~~~

## Results
Different populations of Mario learned in different way and for different environments. Here are some of the things the AI was able to learn:

Mario beating 1-1:
![Mario Beating 1-1](/SMB_level1-1_gen_258_win.gif)

Mario beating 4-1:
![Mario Beating 4-1](/SMB_level4-1_gen_1378_win.gif)

Mario learning to walljump:
![Mario Learning to Walljump](/SMB_level4-1_walljump.gif)