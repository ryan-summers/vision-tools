import argparse
import cv2
import random
import os.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subsamples an image randomly a number of times to generate random sub-images.');
    parser.add_argument('image', help='The image to subsample')
    parser.add_argument('--minw', help='Minimum width of subsamples', default=25)
    parser.add_argument('--minh', help='Minimum height of subsamples', default=25)
    parser.add_argument('--cropl', help='Crop left pixel count', default=300, type=int)
    parser.add_argument('--cropr', help='Crop right pixel count', default=300, type=int)
    parser.add_argument('--cropt', help='Crop top pixel count', default=100, type=int)
    parser.add_argument('--cropb', help='Crop bottom pixel count', default=100, type=int)
    parser.add_argument('--output', help='output directory', type=str)
    parser.add_argument('count', help='The number of subsamples to generate')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    rows, columns = image.shape[:2]
    image = image[args.cropt:rows - args.cropb, args.cropl:columns - args.cropr]
    rows, columns = image.shape[:2]

    if args.output is None:
        dirname = os.path.dirname(args.image)
    else:
        dirname = args.output

    filename = os.path.splitext(os.path.basename(args.image))[0]

    for i in range(0, int(args.count)):
        row_start = random.randint(0, rows - args.minh)
        row_end = random.randint(row_start + args.minh, rows)
        col_start = random.randint(0, columns - args.minw)
        col_end = random.randint(col_start + args.minw, columns)
        sub_image = image[row_start:row_end, col_start:col_end]
        new_rows, new_cols = sub_image.shape[:2]
        if new_rows >= args.minh and new_cols >= args.minw:
            cv2.imwrite('{}/{}-{}.jpg'.format(dirname, filename, i), sub_image)
        else:
            print('Invalid image size! Cols: {} Rows: {} | {}:{}, {}:{}'.format(columns, rows, row_start, row_end, col_start, col_end))
