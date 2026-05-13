from utils.streaming_frames import make_frame_source
from argparse import Namespace

args = Namespace(
    streaming_max_frames=0, 
    tum_frame_stride=1, 
    tum_association_max_dt=0.03, 
    tum_sequence='', 
    streaming_frame_stride=1
)
source = make_frame_source('/data/TUM/rgbd_dataset_freiburg1_desk', args)
print(f"Total frames: {len(source)}")
