# Compatibility shim for older imports. Reader implementations live in scene/readers/.
from scene.readers.common import CameraInfo, SceneInfo, fetchPly, fetchPlyFlexible, getNerfppNorm, storePly
from scene.readers.colmap import readColmapCameras, readColmapSceneInfo
from scene.readers.blender import readCamerasFromTransforms, readNerfSyntheticInfo
from scene.readers.rgbd_sequence import readRGBDSequenceSceneInfo
from scene.readers.tum import readTUMCameras, readTUMSceneInfo
from scene.readers.scannet import readScanNetCameras, readScanNetSceneInfo

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "RGBDSequence": readRGBDSequenceSceneInfo,
    "ScanNet": readScanNetSceneInfo,
    "TUM": readTUMSceneInfo,
}
