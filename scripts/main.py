from pathlib import Path
import toml
import cv2
from utils import CameraParam, LensUndistorter, ImageSaver, LensUndistorterWithKroi

# Load Pathes
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(BASE_DIR, "data")
CFG_PARAM_PATH = str(Path(DATA_DIR, "camera_param.toml"))
RGB_IMAGE_PATH = str(Path(DATA_DIR, "rgb_img.png"))
RESULT_SAVE_DIR = str(Path(BASE_DIR, "results"))
result_saver = ImageSaver(RESULT_SAVE_DIR)

# Get config file and rgb image
dict_param = toml.load(open(CFG_PARAM_PATH))
rgb_img = cv2.imread(RGB_IMAGE_PATH)

# Get camera parameter
camera_param = CameraParam.from_dict(dict_param["Rgb"])
K_rgb = camera_param.intrinsic_matrix
D_rgb = camera_param.distortion
image_height, image_width = camera_param.size

# Image Correction
lens_undistorter = LensUndistorter(K_rgb, D_rgb, image_width, image_height)
lens_undistorter_roi = LensUndistorterWithKroi(K_rgb, D_rgb, image_width, image_height)
rgb_img_undistorted = lens_undistorter.correction(rgb_img)
rgb_img_undistorted_roi = lens_undistorter_roi.correction(rgb_img)


result_saver.save_image("raw_image.png", rgb_img)
result_saver.save_image("rgb_img_undistorted.png", rgb_img_undistorted)
result_saver.save_image("rgb_img_undistorted_roi.png", rgb_img_undistorted_roi)
