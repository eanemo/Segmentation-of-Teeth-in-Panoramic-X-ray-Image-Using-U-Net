import argparse
import tensorflow as tf
import tf2onnx

from model import *


def main(args):
    data_size = (512, 512)

    # Create the model
    model = UNET(input_shape=(data_size[1], data_size[0], 3),
                 last_activation='softmax', num_classes=args.num_classes)

    model.load_weights(args.model_file)

    spec = (tf.TensorSpec(
        (None, data_size[1], data_size[0], 3), tf.float32, name="input"),)

    output_path = args.output_file

    if '.onnx' not in output_path:
        output_path += '.onnx'

    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento de la red UNet.')
    parser.add_argument('model_file', help='Model file path')
    parser.add_argument('output_file', help='Output ONNX model file path')
    parser.add_argument('--num_classes', type=int,
                        help="Número de clases de la segmentación (classes + fondo)", default=2)

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)
