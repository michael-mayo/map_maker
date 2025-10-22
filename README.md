
# Map Generator

This program uses simplex noise to generate map heightmaps suitable for the City Skylines 2 Map Editor.

To set everything up:
* install a python3 virtual environment
* clone the map maker repo
* use pip to install required libraries, `pip install -r requirements.txt`

To run the program using the following command:
```
python map.py --seed [SEED] --it [IT]
```
Replace SEED with your random number seed and IT with the number of maps (iterations) you want.
The program will generate pairs of PNG images, `map[it]_wm.png` and `map[it]_hm.png`
being the world and height maps respectively. These files can be copied to your CS2 heightmaps
folder (`C:\Users\USERNAME\AppData\LocalLow\Colossal Order\Cities Skylines II\Heightmaps` on my PC)
and then imported in the map editor.

## Sample maps

| seed | it | world                            | height                           |
|------|----|----------------------------------|----------------------------------|
| 619  | 1  | ![map](./readme-images/1_wm.png) | ![map](./readme-images/1_hm.png) |
| 619  | 4  | ![map](./readme-images/4_wm.png) | ![map](./readme-images/4_hm.png) |

### Map 1 after loading into the map editor:
![map](./readme-images/1_map_editor.png)
### Map 4 after loading into the map editor:
![map](./readme-images/4_map_editor.png)



