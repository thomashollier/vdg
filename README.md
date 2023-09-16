# vdg

Repo of image processing python scripts with ocv2


## Set up python environment


```
brew install pyenv
pyenv install 3.11.3
cd ~/Documents/slitscans/vdg
mkdir venv && cd venv
pyenv global 3.11.3
python -m venv $(pwd)

# the path to the python command is a link to the
# pyenv version of the binary

source venv/bin/activate 
pip install opencv-python
pip install pycubelut
```

rr

Run python interpreter from the command line rather than from within the script file

(pycubelut will fail because of numpy errors. fix the files and it will work)
