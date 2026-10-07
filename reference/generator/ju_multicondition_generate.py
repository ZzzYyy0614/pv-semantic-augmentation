import argparse, os, sys, glob
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms
flip = transforms.RandomHorizontalFlip(p=0.5)

def make_batch(label, mask, device):
    image = np.array(Image.open(mask).convert("RGB"))
    image = image.astype(np.uint8)
    imag = Image.fromarray(image)
    imge = np.array(imag.resize((256, 256), resample=PIL.Image.BICUBIC)).astype(np.float32)/127.5 - 1.0
    image = imge[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    # mask = np.array(Image.open(mask).convert("RGB"))
    # mask = mask.astype(np.uint8)
    # mask_img = Image.fromarray(mask)
    # mask = np.array(mask_img.resize((256, 256), resample=PIL.Image.BICUBIC)).astype(np.float32)/127.5 - 1.0
    # mask = mask[None,None]
    # mask = torch.from_numpy(mask)

    batch = {"masked_image": image, "class_label": torch.tensor([label])}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        # batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        default = '/disk1/zhuy/PV/EL/masked_EL/test3/',
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default = '/disk1/zhuy/PV/EL/masked_EL/test/',
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*.png")))
    # images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("logs/2024-06-18T16-02-56_ju_multi_condition/configs/2024-06-18T16-02-56-project.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("logs/2024-06-18T16-02-56_ju_multi_condition/checkpoints/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)

    result = Image.new('RGB', (len(masks)*256, 2*256))
    offset = 0
    with torch.no_grad():
        with model.ema_scope():
            # for image, mask in tqdm(zip(images, masks)):
            label_list = [1,0,0,0,1,1,1,0,1]
            for i,mask in tqdm(enumerate(masks)):
                outpath = os.path.join(opt.outdir, os.path.split(mask)[1])
                label = label_list[i]
                batch = make_batch(label, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                c2 = model.cond_stage2_model(batch)
                cond = [c, c2]

                shape = (c.shape[1],)+c.shape[2:]
                # shape = c.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=cond,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                # image = torch.clamp((batch["image"]+1.0)/2.0,
                #                     min=0.0, max=1.0)
                # mask = torch.clamp((batch["mask"]+1.0)/2.0,
                #                    min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)*127.5,
                                              min=0.0, max=255)
                # predicted_image = x_samples_ddim

                # inpainted = (1-mask)*image+mask*predicted_image
                # inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                sav_image = (predicted_image.cpu().numpy()[0]).transpose(1,2,0)
                # Image.fromarray(sav_image.astype(np.uint8)).save(outpath)

                mask = Image.open(mask).convert("L").resize((256,256))
                predict = Image.fromarray(sav_image.astype(np.uint8))
                result.paste(mask, (offset, 0))
                result.paste(predict, (offset, 256))
                offset = offset + 256

            result.save(os.path.join(opt.outdir, 'multi_condition_result2.png'))
