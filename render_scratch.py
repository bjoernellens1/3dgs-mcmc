import torch
import torchvision
from scene import Scene, GsplatGaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def main():
    parser = ArgumentParser()
    model = ModelParams(parser)
    pipe = PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = parser.parse_args(args=["-s", "/data/tum_raw/rgbd_dataset_freiburg1_desk", "-m", "output/ablation_rolling_seed"])
    
    dataset = model.extract(args)
    pipe = pipe.extract(args)
    
    gaussians = GsplatGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration="final")
    
    cam = scene.getTrainCameras()[0]
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg)
        img = pkg["render"]
        gt = cam.original_image
        
    torchvision.utils.save_image(img, "artifacts/rendered_0.png")
    torchvision.utils.save_image(gt, "artifacts/gt_0.png")
    print("Saved images to artifacts/")

if __name__ == "__main__":
    main()
