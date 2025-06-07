# Gui_for_musubi-tuner

core from https://github.com/kohya-ss/musubi-tuner

A user-friendly GUI for the **Kohya_ss musubi-tuner**, designed for easy use!

## Requirements:
1. You will need an **embedded Python system** for this to work, 3.12 works fine!
2. Ensure the **hunyuan Video checkpoints** are located in the `./models` folder.
3. Create an `./output` folder for output files.
4. Install all the dependencies required by Kohya in the `python_embeded` folder.
5. Place the `python_embeded` folder inside the **musubi-tuner** directory.

## Setup:
1. Navigate to the **musubi-tuner** folder.
2. Run the following command to launch the GUI:

   ```bash
   .\python_embeded\python.exe train_gui.py
   ```

The GUI now supports FramePack **F1** mode and the `--one_frame` training option found in the upstream project.
