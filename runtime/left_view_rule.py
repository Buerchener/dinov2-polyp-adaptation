"""Conservative RGB-only left crop candidate; no score or label-dependent fallback."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps




def candidate(record):
    w,h=record['width'],record['height'];left=record['box'][0]
    if record['candidate_status']!='CANDIDATE' or not .05<=left/w<=.45:
        return [0,0,w,h]
    return [left,0,w,h]
