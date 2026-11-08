import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def gauss_add_GaussNoisy(image):
    mean = 0
    sigma = 25 
    [img_height, img_width] = image.shape
    gauss = np.random.normal(mean, sigma, (img_height, img_width))
    noisy_img = image + gauss
    noisy_img_gauss = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img_gauss



def add_salt_pepper_noise(image):
    salt_prob = 0.03
    pepper_prob = 0.03
    noisy_img = np.copy(image)
    height, width = image.shape
    random_matrix = np.random.rand(height, width)
    salt_mask = random_matrix < salt_prob
    noisy_img[salt_mask] = 255
    pepper_mask = (random_matrix >= salt_prob) & (random_matrix < salt_prob + pepper_prob)
    noisy_img[pepper_mask] = 0
    return noisy_img

def Denoise_fft(image):
    f = np.fft.fft2(image)
    fshift = abs(np.fft.fftshift(f))
    f_angle = np.angle(np.fft.fftshift(f))

    f_g = np.log(1 + abs(fshift))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(f_g, cmap='gray')
    plt.title('fft image')

    patch = 50
    row, col = image.shape
    row1, col1 = int(row/2), int(col/2)
    mask = np.zeros((row, col), np.uint8)
    mask[row1-patch:row1+patch, col1-patch:col1+patch] = 1
    fshift = fshift*mask
    
    f_g = np.log(1 + abs(fshift))
    plt.subplot(1, 3, 2)
    plt.imshow(f_g, cmap='gray')
    plt.title('ifft map')

    s1_real = fshift * np.cos(f_angle)
    s1_imag = fshift * np.sin(f_angle)
    s2 = np.zeros(image.shape, dtype=complex)
    s2.real = np.array(s1_real)
    s2.imag = np.array(s1_imag)
    
    ishift = np.fft.ifftshift(s2)
    iimage = np.fft.ifft2(ishift)
    iimage = np.abs(iimage)
    
    plt.subplot(1, 3, 3)
    plt.imshow(iimage, cmap='gray')
    plt.title('ifft image')
    

if __name__ == '__main__':
    
    image = cv.imread('1.png')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('original image')

    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    noisy_img_gauss = gauss_add_GaussNoisy(image)
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.uint8(noisy_img_gauss), cmap='gray', vmin=0, vmax=255)
    plt.title('Gauss_noisy_image')

    noisy_img_salt = add_salt_pepper_noise(image)
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.uint8(noisy_img_salt), cmap='gray', vmin=0, vmax=255)
    plt.title('Salt_noisy_image')

    h = 3
    image_denoise_mean = cv.blur(np.uint8(noisy_img_gauss), (h, h))
        
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.uint8(image_denoise_mean), cmap='gray', vmin=0, vmax=255)
    plt.title('Denoise_mean_3')
    
    h = 9
    image_denoise_mean = cv.blur(np.uint8(noisy_img_gauss), (h, h))

    plt.subplot(1, 2, 2)
    plt.imshow(np.uint8(image_denoise_mean), cmap='gray', vmin=0, vmax=255)
    plt.title('Denoise_mean_9')

    image_denoise_gauss = cv.GaussianBlur(np.uint8(noisy_img_gauss), (5, 5), 1)
                                           
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.uint8(image_denoise_gauss), cmap='gray', vmin=0, vmax=255)
    plt.title('Denoise_gauss_1')
    
    image_denoise_gauss = cv.GaussianBlur(np.uint8(noisy_img_gauss), (5, 5), 5)
                                           
    plt.subplot(1, 2, 2)
    plt.imshow(np.uint8(image_denoise_gauss), cmap='gray', vmin=0, vmax=255)
    plt.title('Denoise_gauss_5')

    Denoise_fft(noisy_img_gauss)
    
    
    plt.show()



