"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
from glob import glob
import SimpleITK
import torch
import numpy as np
import gc, os
import time
from resources.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from resources.nnunetv2.paths import nnUNet_results, nnUNet_raw
from resources.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join

print('All required modules are loaded!!!')

# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# RESOURCE_PATH = Path("resources")

INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("./test/output")
RESOURCE_PATH = Path("resources")

nnUNet_raw = INPUT_PATH / "images/ct-angiography"
nnUNet_results = OUTPUT_PATH / "images/aortic-branches"
os.makedirs(nnUNet_results, exist_ok=True)
nnUNet_source = RESOURCE_PATH


def run():
    _show_torch_cuda_info()
    # Read the input
    os.environ['nnUNet_compile'] = 'F' 
    ct_mha = subfiles(nnUNet_raw, suffix='.mha')[0]
    uuid = os.path.basename(os.path.splitext(ct_mha)[0])
    output_file_path = os.path.join(nnUNet_results, uuid)

    print(output_file_path)
    # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)
    
    checkpoint_name='checkpoint_final.pth'
    model_path = os.path.join(nnUNet_source, 'Dataset919_CTA/nnUNetTrainer_NoMirroring_ep1000_finetuning__nnUNetResEncUNetLPlans__3d_fullres')

    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds='all',
        checkpoint_name=checkpoint_name,
    )

    predictor.dataset_json['file_ending'] = '.mha'
    images, properties = SimpleITKIO().read_images([ct_mha])
    iterator = predictor.get_data_iterator_from_raw_npy_data([images], None, [properties], None, 3)
    result = predictor.predict_from_data_iterator(iterator, False, 3)

    pred_array = result[0]
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)

    image = SimpleITK.GetImageFromArray(pred_array)
    image.SetDirection(properties['sitk_stuff']['direction'])
    image.SetOrigin(properties['sitk_stuff']['origin'])
    image.SetSpacing(properties['sitk_stuff']['spacing'])
    SimpleITK.WriteImage(
        image,
        output_file_path + '.mha',
        useCompression=True,
    )

    print('Prediction finished')
    # Process the inputs: any way you'd like
    ###############################################################
    # Set the environment variable to handle memory fragmentation
    torch.cuda.empty_cache()
   
    ######################### MY LINES END ################################################
    # Save mha output
    print('Saved!!!')
    return 0



def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())



