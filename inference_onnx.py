import onnxruntime
import numpy as np
import cv2
import argparse

def main(args):
    data_size = (512, 512)

    # Read the image
    img = cv2.imread(args.image_file)
    # Normalize
    img = img / 255.0
    # Resize
    img = cv2.resize(img, data_size)
    # Prepare for model input size
    input_img = np.expand_dims(img, axis=0).astype(np.float32)
    print(input_img.shape)

    sess = onnxruntime.InferenceSession(args.model_file)
    # Run the inference
    result = sess.run(None, {'input': input_img})
    print(result[0].shape)
    output_img = np.argmax(result[0], axis=3).astype(np.uint8) * 255
    output_img = np.squeeze(output_img)
    
    print(output_img.shape)
    cv2.imwrite("./output.png", output_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento de la red UNet.')
    parser.add_argument('model_file', help='Model file path')
    parser.add_argument('image_file', help='Input image file path')
    parser.add_argument('--num_classes', type=int,
                        help="Número de clases de la segmentación (classes + fondo)", default=2)

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)
