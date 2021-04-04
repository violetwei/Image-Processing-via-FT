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

def denoise(fft_img, low_freq = 0, high_freq = 0.15):
    #fft_img = img.copy()
    fft_img = np.asarray(fft_img, dtype=complex)
    n, m = fft_img.shape

    fft_img[int(n * low_freq):-int(n * high_freq), :] = 0
    fft_img[:, int(m * low_freq):-int(m * high_freq)] = 0

    # count the number of non-zeros
    nonzero_count = np.count_nonzero(fft_img)
    print("The number of non-zeros using: ", nonzero_count)
    print("The fraction they represent of the original Fourier coefficients: ", nonzero_count / fft_img.size)

    # Denoising the image using 2d fft inverse
    denoised_img = twoD_FFT_inverse(fft_img).real
    return denoised_img

def compress(fft_img, compression_level):
    compress_img = np.asarray(fft_img, dtype=complex)

    m = int((compress_img.shape[0] / 2) * math.sqrt(1 - (compression_level / 100)))
    n = int((compress_img.shape[1] / 2) * math.sqrt(1 - (compression_level / 100)))

    compress_img[m:-n, :] = 0+0.j
    compress_img[:, n:-m] = 0+0.j

    precompress_nonzeros_count = np.count_nonzero(fft_img)
    postcompress_nonzeros_count = np.count_nonzero(compress_img)

    print("The non-zero values for compression level {}% are {}".format(compression_level, postcompress_nonzeros_count))

    save_npz('moonlanding-{}-compression.csr'.format(compression_level),
             csr_matrix(compress_img))

    # inverse transform the modified Fourier coefficients to obtain the image
    compressed = twoD_FFT_inverse(csr_matrix(compress_img).toarray())
    return compressed

def mode1(input_image):
    # fast mode
    print("Mode [1] Selected")
    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # plot image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [1]: Fast Mode')
    ax1.set_title('Original Input Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('2D FFT Logarithmic Colormap')
    ax2.imshow(np.abs(img_2dfft), norm=colors.LogNorm())
    plt.show()

def mode2(input_image):
    # denoise
    print("Mode [2] Selected")

    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # denoising
    denoised_img = denoise(img_2dfft)

    # plot image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [2]: Denoising Mode')
    ax1.set_title('Original Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('Denoised Image')
    ax2.imshow(denoised_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    plt.show()


def mode3(input_image):
    print("Mode [3] Selected")

    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # plot images
    fig, ax = plt.subplots(2, 3) # 2 by 3
    fig.suptitle('Mode [3]: Compressing Mode')

    # compression level: 0%
    compressed_img1 = compress(img_2dfft, 0)
    ax[0, 0].set_title('0% Compression')
    #ax[0, 0].imshow(compressed_img1, plt.cm.gray)
    ax[0, 0].imshow(np.real(compressed_img1)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

    # compression level: 20%
    compressed_img2 = compress(img_2dfft, 20)
    ax[0, 1].set_title('20% Compression')
    #ax[0, 1].imshow(compressed_img2, plt.cm.gray)
    ax[0, 1].imshow(np.real(compressed_img2)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

    # compression level: 35%
    compressed_img3 = compress(img_2dfft, 35)
    ax[0, 2].set_title('35% Compression')
    #ax[0, 2].imshow(compressed_img3, plt.cm.gray)
    ax[0, 2].imshow(np.real(compressed_img3)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

    # compression level: 55%
    compressed_img4 = compress(img_2dfft, 55)
    ax[1, 0].set_title('55% Compression')
    #ax[1, 0].imshow(compressed_img4, plt.cm.gray)
    ax[1, 0].imshow(np.real(compressed_img4)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

    # compression level: 75%
    compressed_img5 = compress(img_2dfft, 75)
    ax[1, 1].set_title('75% Compression')
    #ax[1, 1].imshow(compressed_img5, plt.cm.gray)
    ax[1, 1].imshow(np.real(compressed_img5)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

    # compression level: 95%
    compressed_img6 = compress(img_2dfft, 95)
    ax[1, 2].set_title('95% Compression')
    #ax[1, 2].imshow(compressed_img6, plt.cm.gray)
    ax[1, 2].imshow(np.real(compressed_img6)[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)

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
        # Fast Mode
        mode1(input_image)
    elif input_mode == 2:
        # Denoising
        mode2(input_image)
    elif input_mode == 3:
        mode3(input_image)


if __name__ == '__main__':
    __main__()
