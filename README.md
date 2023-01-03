# vdg

Repo of image processing python scripts with ocv2


## Set up python environment


```
brew install pyenv
pyenv install 3.9.16
cd ~/Documents/slitscans/vdg
mkdir venv && cd venv
pyenv global 3.9.16
python -m venv $(pwd)

# the path to the python command is a link to the
# pyenv version of the binary

source venv/bin/activate 
pip install opencv-python
```

Run python interpreter from the command line rather than from within the script file

