import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )




np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    return img
    ax.imshow(img)







from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator_1 = SAM2AutomaticMaskGenerator(sam2)
# masks = mask_generator.generate(image)


mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=96,
    points_per_batch=512,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=2,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

mask_generator = mask_generator_1

def human_server(input_image):
    image = np.array(input_image.convert("RGB"))
    masks = mask_generator.generate(image)
    return show_anns(masks)

def launch_human_server():
    hface = gr.Interface(
        fn=human_server,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.Image(type="numpy", label="Generated Masks"),
        title="SAM2 Mask Generator",
        description="Upload an image to generate a list of segmentation masks using SAM2."
    )
    hface.launch(server_port=7861)
    exit()

def launch_np_server():

    def generate_masks(input_image):
        image = np.array(input_image.convert("RGB"))
        masks = mask_generator.generate(image)
        mask_images = [ann['segmentation'].astype(np.uint8) * 255 for ann in masks]
        return mask_images

    iface = gr.Interface(
        fn=generate_masks,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.Gallery(type="numpy", label="Generated Masks"),
        title="SAM2 Mask Generator",
        description="Upload an image to generate a list of segmentation masks using SAM2."
    )

    iface.launch(server_port=7861)

def demo():

    image = Image.open('../Depth-Anything-V2/depth/assets/examples/demo01.jpg')
    image = np.array(image.convert("RGB"))
    masks2 = mask_generator_2.generate(image)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show()
    plt.savefig('output.png')
    plt.close()

def launch_raw_server():
    def generate_masks_raw(input_image):
        """
        Generate segmentation masks for the input image and return the raw mask data.
        
        Args:
            input_image (PIL.Image): The input image.
        
        Returns:
            list: A list of dictionaries containing the mask data with 'segmentation' converted to lists.
        """
        image = np.array(input_image.convert("RGB"))
        masks = mask_generator.generate(image)
        for mask in masks:
            if 'segmentation' in mask:
                mask['segmentation'] = mask['segmentation'].tolist()
        return masks

    iface = gr.Interface(
        fn=generate_masks_raw,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.JSON(label="Generated Masks"),
        title="SAM2 Mask Generator (Raw)",
        description="Upload an image to generate a list of segmentation masks using SAM2 and return the raw mask data."
    )
    iface.launch(server_port=7861)