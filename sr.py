import os

import cv2
import tqdm
from cv2 import dnn_superres


class Model:
    def __init__(self, path, name, scale) -> None:
        self.path = path
        self.name = name
        self.scale = scale

        self.model = dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(path)
        self.model.setModel(name, scale)


Model.EDSR_x4 = Model('models/EDSR_x4.pb', 'edsr', 4)
Model.ESPCN_x4 = Model('models/ESPCN_x4.pb', 'espcn', 4)
Model.LapSRN_x8 = Model('models/LapSRN_x8.pb', 'lapsrn', 8)


def upscale_video(input_path, output_path, model, show_progressbar=False):
    reader = cv2.VideoCapture(input_path)

    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) * model.scale)
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) * model.scale)

    video_fourcc = _infer_fourcc(output_path)
    if video_fourcc is None:
        return False

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *video_fourcc), fps, (width, height))

    if show_progressbar:
        n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm.tqdm(total=n_frames)

    success, frame = reader.read()
    while success:
        upscaled_frame = _upscale_image(frame, model)
        writer.write(upscaled_frame)
        success, frame = reader.read()

        if show_progressbar:
            pbar.update()

    if show_progressbar:
        pbar.close()

    reader.release()
    writer.release()


def upscale_image(input_path, output_path, model):
    input_img = cv2.imread(input_path)
    upscaled_img = _upscale_image(input_img, model)
    cv2.imwrite(output_path, upscaled_img)


def _infer_fourcc(filename):
    fourcc_dict = {
        '.mp4': 'mp4v',
        '.avi': 'RGBA',
    }

    _, ext = os.path.splitext(filename.lower())

    return fourcc_dict.get(ext, None)


def _upscale_image(image, model):
    return model.model.upsample(image)


if __name__ == '__main__':
    import argparse
    import sys

    video_exts = ['.mp4', '.avi']
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    def _ext_in_list(path, exts):
        _, ext = os.path.splitext(path.lower())
        return ext in exts

    def is_video(path):
        return _ext_in_list(path, video_exts)

    def is_image(path):
        return _ext_in_list(path, image_exts)

    models = {
        'edsr': Model.EDSR_x4,
        'espcn': Model.ESPCN_x4,
        'lapsrn': Model.LapSRN_x8,
    }

    arg_parser = argparse.ArgumentParser(sys.argv[0])
    arg_parser.add_argument('-i', '--input', action='store',
                            required=True, type=str, help='Input filename path')
    arg_parser.add_argument('-o', '--output', action='store',
                            required=True, type=str, help='Output filename path')
    arg_parser.add_argument('-m', '--model', action='store', required=True,
                            type=str, help='The name of the model to be used for the upscaling')

    args = arg_parser.parse_args(sys.argv[1:])

    model = models.get(args.model, None)

    if model is None:
        print(f'Error: unknown model {args.model}')

    if is_video(args.input) and is_video(args.output):
        upscale_video(args.input, args.output, model, show_progressbar=True)

    elif is_image(args.input) and is_image(args.output):
        upscale_image(args.input, args.output, model)

    else:
        print('Error: input/output file is neither a video nor an image')
