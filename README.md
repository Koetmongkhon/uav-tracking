# Senior-Project

1. install pytorch with gpu https://pytorch.org/get-started/locally/ or with command (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117)
2. clone yolov5 (https://github.com/ultralytics/yolov5) to use yolov5 in detecting step.
3. clone pysot (https://github.com/STVIR/pysot) to use SiamRPN in tracking step.
4. Export PYTHONPATH if you work with Linux you can run "export PYTHONPATH=/path/to/pysot:$PYTHONPATH" in your terminal but if you work with Windows you can run "set PYTHONPATH=/path/to/pysot" in your command prompt
5. run command "pip install -r requirement.txt" in your terminal/command prompt
6. if you want to use Gstreamer with OpenCV you need to install OpenCV manually. You can follow https://youtu.be/NvUd2EjndLI. But if you don't need Gstreamer to work with OpenCV you can just simply run "pip install opencv-python opencv-contrib-python" in your terminal/command prompt.

# run program
webcam as input: python main_program.py or python main_program.py -s webcam
video as input: python main_program.py -s video -i {path to video}
save output to file(mp4): python main_program.py -s webcam -o {output file name} or python main_program.py -s video -i {path to video} -o {output file name}