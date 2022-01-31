import argparse
import shutil
from glob import glob
from os.path import join, split
from os import makedirs
import Augmentor
import pip


def main(args):
    images_path = args['images_path']
    masks_path = args['masks_path']
    output_path = args['output_path']
    samples = args['augment_to']
    print(images_path)

    # The augmented dataset should be a randomised image with some random distortion and maybe some brightness added/substracted
    pipeline = Augmentor.Pipeline(
        source_directory=images_path, output_directory=output_path)
    pipeline.ground_truth(masks_path)
    pipeline.random_distortion(
        probability=1, grid_width=4, grid_height=4, magnitude=8)
    pipeline.random_brightness(0.4, 0.9, 1.1)

    if samples > 0:
        pipeline.sample(samples)

        # Copy images to destination
        src_dir = output_path
        dst_img_dir = join(output_path, "images")
        dst_gt_dir = join(output_path, "masks")

        makedirs(dst_img_dir, exist_ok=True)
        makedirs(dst_gt_dir, exist_ok=True)

        files = glob(src_dir + "/*.png")

        img_remove_label = 'images_png_original_'
        gt_remove_label = '_groundtruth_(1)_images_png_'

        for filepath in files:
            _, filename = split(filepath)

            if gt_remove_label in filename:
                dst_file = join(
                    dst_gt_dir, filename.replace(gt_remove_label, ''))
            else:
                dst_file = join(
                    dst_img_dir, filename.replace(img_remove_label, ''))

            #print("Copying {} to {} ...".format(filepath, dst_file))
            shutil.move(filepath, dst_file)

    else:
        print("Error: --augment-to param must be greater than 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse requirements and model paths.')
    parser.add_argument(
        '--images-path', help='Path to the original images', required=True, dest='images_path')
    parser.add_argument(
        '--masks-path', help='Path to the original mask images', required=True, dest='masks_path')
    parser.add_argument('--output-path', dest='output_path',
                        help='Path to the output dir', required=True)
    parser.add_argument('--augment-to', dest='augment_to',
                        help='Number of the augmented dataset', type=int, required=True)
    args = vars(parser.parse_args())

    main(args)
