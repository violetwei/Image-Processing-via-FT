import numpy as np
import cv2 as cv
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import math
import argparse
import time
import statistics

def resize(n):
    power = int(math.log(n, 2))
    new_size = int(pow(2, power+1))
    return new_size


# pad the image so that it has a length or width that is a power of 2
def pad_image(raw_image):
    original_shape = raw_image.shape
    new_shape = resize(original_shape[0]), resize(original_shape[1])
    img = np.zeros(new_shape)
    img[:original_shape[0], :original_shape[1]] = raw_image
    return img

def oneD_slowFT(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    ans = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            ans[k] += np.exp(-2j * np.pi * k * n / N) * img[n]
    return ans

def oneD_slowFT_inverse(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    ans = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            ans[n] += np.exp(2j * np.pi * k * n / N) * img[k]
        ans[n] /= N
    return ans

def oneD_FFT(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    if N % 2 != 0:
        raise AssertionError("ERROR! The size of img is supposed to be power of 2.")
    elif N > 16:
        # call function recursively
        odd = oneD_FFT(img[1::2])
        even = oneD_FFT(img[::2])

        ans = np.zeros(N, dtype=complex)
        half_N = N // 2
        for n in range(N):
            ans[n] = even[n % half_N] + \
                       np.exp(-2j * np.pi * n / N) * odd[n % half_N]
        return ans
    else: #N <= 16
        return oneD_slowFT(img)

def oneD_FFT_inverse(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    if N % 2 != 0:
        raise AssertionError("ERROR! The size of img is supposed to be power of 2.")
    elif N > 16:
        # call function recursively
        odd = oneD_FFT_inverse(img[1::2])
        even = oneD_FFT_inverse(img[::2])

        ans = np.zeros(N, dtype=complex)
        half_N = N // 2
        for n in range(N):
            ans[n] = even[n % half_N] * half_N + \
                       half_N * np.exp(2j * np.pi * n / N) * odd[n % half_N]
            ans[n] /= N
        return ans
    else: #N <= 16
        return oneD_slowFT_inverse(img)


def twoD_FFT(img):
    img = np.asarray(img, dtype=complex)
    n, m = img.shape
    ans = np.zeros(
        (n, m),
        dtype=complex
    )
    for c in range(m):
        ans[:, c] = oneD_FFT(img[:, c])
    for r in range(n):
        ans[r, :] = oneD_FFT(ans[r, :])
    return ans

def twoD_FFT_inverse(img):
    img = np.asarray(img, dtype=complex)
    n, m = img.shape
    ans = np.zeros(
        (n, m),
        dtype=complex
    )
    for r in range(n):
        ans[r, :] = oneD_FFT_inverse(img[r, :])
    for c in range(m):
        ans[:, c] = oneD_FFT_inverse(ans[:, c])
    return ans

def mode1(input_image):
    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # plot img
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [1]: Fast Mode')
    ax1.set_title('Original Input Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('2D FFT Logarithmic Colormap')
    ax2.imshow(np.abs(img_2dfft), norm=colors.LogNorm())
    plt.show()

def __main__():
    # parse arguments from inputs
    parser = argparse.ArgumentParser(description='fft parser')

    parser.add_argument('-m',
                        type=int,
                        default=1,
                        dest='mode',
                        help='fft modes: [1]->fast mode; [2]->denoising mode; [3]->compressing & saving mode; [4]->plotting mode')

    parser.add_argument('-i',
                        type=str,
                        default='moonlanding.png',
                        dest='image',
                        help='fft target image')

    input_args = parser.parse_args()
    input_mode = input_args.mode
    input_image = input_args.image

    if input_mode == 1:
        mode1(input_image)



if __name__ == '__main__':
    __main__()
