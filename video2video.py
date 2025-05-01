"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Image to ASCII")
    parser.add_argument("--input", type=str, default="data/input.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="data/output.mp4", help="Path to output video")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "complex"],
                        help="10 or 70 different characters")
    parser.add_argument("--background", type=str, default="white", choices=["black", "white"],
                        help="background's color")
    parser.add_argument("--num_cols", type=int, default=100, help="number of character for output's width")
    parser.add_argument("--scale", type=int, default=1, help="upsize output")
    parser.add_argument("--fps", type=int, default=0, help="frame per second")
    parser.add_argument("--overlay_ratio", type=float, default=0.2, help="Overlay width ratio")
    args = parser.parse_args()
    return args


def main(opt):
    if opt.mode == "simple":
        CHAR_LIST = '@%#*+=-:. '
    else:
        CHAR_LIST = r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    if opt.background == "white":
        bg_code = 255
    else:
        bg_code = 0
    font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=int(10 * opt.scale))

    cap = cv2.VideoCapture(opt.input)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {opt.input}")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_length <= 0:
        raise ValueError(f"Invalid video length: {video_length} frames")

    if opt.fps == 0:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"FPS: {fps}")
    else:
        fps = opt.fps
    num_chars = len(CHAR_LIST)
    num_cols = opt.num_cols
    progress_bar = tqdm(total=video_length, desc="Processing frames", unit="frame")

    crop_bbox = None
    out = None 
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag: 
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        cell_width = width / opt.num_cols
        cell_height = 2 * cell_width
        num_rows = int(height / cell_height)
        if num_cols > width or num_rows > height:
            print("Too many columns or rows. Use default setting")
            cell_width = 6
            cell_height = 12
            num_cols = int(width / cell_width)
            num_rows = int(height / cell_height)

        char_bbox = font.getbbox("A")
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3]
        out_width = char_width * num_cols
        out_height = 2 * char_height * num_rows
        # print(f"Output size: {out_width}x{out_height}")
        out_image = Image.new("L", (out_width, out_height), bg_code)

        
        draw = ImageDraw.Draw(out_image)
        for i in range(num_rows):
            line = "".join([CHAR_LIST[min(int(np.mean(
                image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                int(j * cell_width):min(int((j + 1) * cell_width), width)]) * num_chars / 255), num_chars - 1)] for j in
                            range(num_cols)]) + "\n"
            draw.text((0, i * char_height), line, fill=255 - bg_code, font=font)

        if crop_bbox is None:
            if opt.background == "white":
                crop_bbox = ImageOps.invert(out_image).getbbox()
            else:
                crop_bbox = out_image.getbbox()

        out_image = out_image.crop(crop_bbox)
        out_image = cv2.cvtColor(np.array(out_image), cv2.COLOR_GRAY2BGR)
        out_image = np.array(out_image)
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(opt.output, fourcc, fps, 
                                (out_image.shape[1], out_image.shape[0]))

        out.write(out_image)

        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    if out:
        out.release()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
