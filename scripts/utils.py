import cv2
import numpy as np
from pathlib import Path
from attr import dataclass, fields


class FromDict:
    @classmethod
    def from_dict(cls, _dict):
        init_kwargs = {}
        for field in fields(cls):
            field_value = _dict[field.name] if field.name in _dict.keys(
            ) else None
            init_kwargs[field.name] = field_value
        return cls(**init_kwargs)


@dataclass(frozen=True, slots=True)
class CameraParam(FromDict):
    width: int = None
    height: int = None
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    k1: float = None
    k2: float = None
    k3: float = None
    k4: float = None

    @property
    def k(self):
        return self.fx, 0., self.cx, 0., self.fy, self.cy, 0., 0., 1.

    @property
    def intrinsic(self):
        return self.fx, 0., self.cx, 0., 0., self.fy, self.cy, 0., 0., 0., 1., 0.

    @property
    def intrinsic_matrix(self):
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])

    @property
    def distortion(self):
        return np.array([self.k1, self.k2, self.k3, self.k4])

    @property
    def size(self):
        return self.height, self.width

    @property
    def center(self):
        return self.cx, self.cy

    @property
    def focal(self):
        return self.fx, self.fy


class LensUndistorter:
    def __init__(self, K_rgb, distortion_params, image_width, image_height):
        self.K_rgb = K_rgb
        self.distortion_params = distortion_params
        self.DIM = (image_width, image_height)
        _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_rgb, self.distortion_params, np.eye(3), self.K_rgb, self.DIM, cv2.CV_16SC2)
        self.map1 = _map1
        self.map2 = _map2
        self.K_rgb_roi = cv2.getOptimalNewCameraMatrix(
            self.K_rgb, self.distortion_params, (image_width, image_height), 0
        )[0]
        self.P_rgb = (self.K_rgb_roi[0][0], 0., self.K_rgb_roi[0][2], 0.,
                      0., self.K_rgb_roi[1][1], self.K_rgb_roi[1][2], 0.,
                      0., 0., 1., 0.)

    def correction(self, image):
        return cv2.remap(
            image, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    @property
    def K_roi(self):
        return self.K_rgb_roi

    @property
    def K(self):
        return self.K_rgb

    @property
    def P(self):
        return self.P_rgb


class LensUndistorterWithKroi:
    def __init__(self, K_rgb, distortion_params, image_width, image_height):
        self.K_rgb = K_rgb
        self.distortion_params = distortion_params
        self.DIM = (image_width, image_height)
        self.K_rgb_roi = cv2.getOptimalNewCameraMatrix(
            self.K_rgb, self.distortion_params, (image_width, image_height), 0
        )[0]
        self.P_rgb = (self.K_rgb_roi[0][0], 0., self.K_rgb_roi[0][2], 0.,
                      0., self.K_rgb_roi[1][1], self.K_rgb_roi[1][2], 0.,
                      0., 0., 1., 0.)
        _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_rgb, self.distortion_params, np.eye(3), self.K_rgb_roi, self.DIM, cv2.CV_16SC2)
        self.map1 = _map1
        self.map2 = _map2

    def correction(self, image):
        return cv2.remap(
            image, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    @property
    def K_roi(self):
        return self.K_rgb_roi

    @property
    def K(self):
        return self.K_rgb

    @property
    def P(self):
        return self.P_rgb


class ImageSaver:
    def __init__(self, save_img_dir):
        self.save_img_dir = save_img_dir

    def save_image(self, name, image):
        cv2.imwrite(
            str(Path(self.save_img_dir, name)),
            image
        )
        cv2.waitKey(10)


