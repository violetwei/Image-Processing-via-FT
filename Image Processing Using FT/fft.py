import numpy as np
import cv2 as cv
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import math
import argparse
import time
import statistics

# Helper function - Resizing
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


# 1D DFT
def oneD_slowFT(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    ans = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            ans[k] += np.exp(-2j * np.pi * k * n / N) * img[n]
    return ans


# Inverse of 1D DFT
def oneD_slowFT_inverse(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    ans = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            ans[n] += np.exp(2j * np.pi * k * n / N) * img[k]
        ans[n] /= N
    return ans


# 1D FFT
def oneD_FFT(img):
    img = np.asarray(img, dtype=complex)
    N = img.shape[0] #size
    if N % 2 != 0:
        raise AssertionError("ERROR! The size of img is supposed to be power of 2.")
    if N > 16:
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


# Inverse of 1D FFT
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
    else: # N <= 16
        return oneD_slowFT_inverse(img)


# 2D FFT
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


# Inverse of 2D FFT
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


# 2D DFT
def twoD_DFT(img):
    img = np.asarray(img, dtype=complex)
    n, m = img.shape
    ans = np.zeros((n, m), dtype=complex)
    for k in range(n):
        for l in range(m):
            for mi in range(m):
                for ni in range(n):
                    ans[k, l] += img[ni, mi] * np.exp(-2j * np.pi * ((l * mi / m) + (k * ni / n)))
    return ans


# Inverse of 2D DFT
def twoD_DFT_inverse(img):
    img = np.asarray(img, dtype=complex)
    n, m = img.shape
    ans = np.zeros((n, m), dtype=complex)
    for k in range(n):
        for l in range(m):
            for mi in range(m):
                for ni in range(n):
                    ans[k, l] += img[ni, mi] * np.exp(2j * np.pi * ((l * mi / m) + (k * ni / n)))
            ans[k, l] /= n * m
    return ans


# Denoising Function
def denoise(fft_img, low_freq, high_freq):
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


# Compressing Function
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


# Mode[1] - Fast Mode
def mode1(input_image):
    # fast mode
    print("Mode [1] Selected")
    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # plot image - generated using the self-implemented algorithms
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [1]: Fast Mode')
    ax1.set_title('Original Input Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('2D FFT Logarithmic Colormap')
    ax2.imshow(np.abs(img_2dfft), norm=colors.LogNorm())
    plt.show()

    # plot image - generated using the built-in np.fft.fft2 function
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [1]: Fast Mode')
    ax1.set_title('Original Input Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('2D FFT Logarithmic Colormap Using np.fft')
    ax2.imshow(np.abs(np.fft.fft2(pad_img)), norm=colors.LogNorm())
    plt.show()


# Mode[2] - Denoising Mode
def mode2(input_image):
    # denoise
    print("Mode [2] Selected")

    # read input image
    raw_image = cv.imread(input_image, cv.IMREAD_GRAYSCALE).astype(float)

    # pad image
    pad_img = pad_image(raw_image)

    # two-dimensional Fourier transform using a fast Fourier transform algorithm
    img_2dfft = twoD_FFT(pad_img)

    # low & high freq threshold parameters
    low = 2.2
    high = 2.8

    # denoising
    denoised_img = denoise(img_2dfft, low, high)
    print('Mode [2] is denoising image with low freq threshold of {} and high freq threshold of {}'.format(low, high))

    # plot image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [2]: Denoising Mode')
    ax1.set_title('Original Image')
    ax1.imshow(pad_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    ax2.set_title('Denoised Image - Low: {} High: {}'.format(low, high))
    ax2.imshow(denoised_img[:raw_image.shape[0], :raw_image.shape[1]], plt.cm.gray)
    plt.show()


# Mode[3] - Compressing Mode
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


# Mode[4] - Plotting Mode
def mode4(input_image):
    print("Mode [4] Selected")

    # re-running the experiment at least 10 times
    number_of_run = 10

    # 2D arrays of random elements of various sizes start from 2^5 and move up to 2^10
    problem_size = 2**4

    # problem size: 2^5=32, 2^6=64, 2^7=128, 2^8=256, 2^9=512
    size_list = [32, 64, 128, 256]

    dft_std = [] # naive method
    dft_mean = [] # naive method
    fft_std = []
    fft_mean = []

    for i in range(len(size_list)):
        fft_runtime = []

        for j in range(number_of_run):
            #dim = np.random.rand(int(math.sqrt(problem_size)), int(math.sqrt(problem_size)))
            dim = np.random.rand(problem_size, problem_size)
            img = pad_image(dim)
            #print('array size: ', len(img))
            start_time = time.perf_counter()
            twoD_FFT(img)
            end_time = time.perf_counter()
            runtime = end_time - start_time
            fft_runtime.append(runtime)
            print('Problem size of {} - FFT Run: {}'.format(len(img), j))

        fft_mean.append(statistics.mean(fft_runtime))
        fft_std.append(statistics.stdev(fft_runtime))
        print('FFT Stats for problem size of {} with {} running times: Mean={}, StdDev={}'.format(
            len(img), number_of_run, statistics.mean(fft_runtime), statistics.stdev(fft_runtime)))

        problem_size *= 2

    print('DFT start')

    # reinitialize size to 32
    problem_size = 2 ** 3

    for i in range(len(size_list)):
        dft_runtime = []

        for j in range(number_of_run):
            #dim = np.random.rand(int(math.sqrt(problem_size)), int(math.sqrt(problem_size)))
            dim = np.random.rand(problem_size, problem_size)
            img = pad_image(dim)
            start_time = time.perf_counter()
            twoD_DFT(img)
            end_time = time.perf_counter()
            runtime = end_time - start_time
            dft_runtime.append(runtime)
            print('Problem size of {} - DFT Run: {}'.format(size_list[i], j))

        dft_mean.append(statistics.mean(dft_runtime))
        dft_std.append(statistics.stdev(dft_runtime))
        print('DFT Stats for problem size of {} with {} running times: Mean={}, StdDev={}'.format(
            size_list[i], number_of_run, statistics.mean(dft_runtime), statistics.stdev(dft_runtime)))

        problem_size *= 2

    # plotting
    # First Plot Graph
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mode [4]: Plotting Mode')

    ax1.set_title('Mean Runtime')
    ax1.set_xlabel('Problem Size')
    ax1.set_ylabel('Runtime Mean(secs)')
    ax1.plot(size_list, dft_mean, label='DFT(naive)')
    ax1.plot(size_list, fft_mean, label='FFT')
    ax1.legend()

    ax2.set_title('StdDev Runtime')
    ax2.set_xlabel('Problem Size')
    ax2.set_ylabel('Runtime StdDev(secs)')
    ax2.plot(size_list, dft_std, label='DFT(naive)')
    ax2.plot(size_list, fft_std, label='FFT')
    ax2.legend()
    plt.show()

    # Second Plot Graph
    plt.errorbar(size_list, dft_mean, yerr=dft_std, fmt='-o', label='DFT')
    plt.errorbar(size_list, fft_mean, yerr=fft_std, fmt='-o', label='FFT')
    plt.title('Runtime Graph of Fourrier Transforms')
    plt.xlabel('Problem Size')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
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
        # Denoising Mode
        mode2(input_image)
    elif input_mode == 3:
        # Compressing Mode
        mode3(input_image)
    elif input_mode == 4:
        # Plotting Mode
        mode4(input_image)

if __name__ == '__main__':
    __main__()
