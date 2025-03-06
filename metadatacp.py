import sys
import subprocess

def getMetadata(f):
    mtdt = {"subject+":"TimeCollapse", "XMP:Creator":"Thomas Hollier", "copyright":"©2025 Relentless Play. All rights reserved", "Software":" ".join(sys.argv)}
    for d in ["GPSPosition", "CreateDate"]:
        cmd = (f'exiftool -s -s -s -c "%+7f" -{d}').split()
        cmd.append(f)
        
        r = subprocess.run(cmd,capture_output=True)
        r = r.stdout.decode("utf-8").replace('"','').rstrip()
        mtdt[d]=r
    return mtdt
    
def setMetadata(f, mtdt):
    options = ["exiftool", "-overwrite_original"]
    for k, v in mtdt.items():
        if k == "GPSPosition":
            k = "composite:GPSPosition"
        options.append(f'-{k}="{v}"')
    options.append(f)
    cmd = options
    print("\n\nCopying metadata to file:")
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True)


