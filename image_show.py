import matplotlib.pyplot as plt


def imageShow(image, is_gray=False):
    # if you wanted to show a single color channel image called 'gray',
    # for example, call as plt.imshow(gray, cmap='gray')

    plt.figure(figsize=(18, 11))
    if is_gray:
        plt.imshow(image, aspect='auto', cmap='gray_r')
    else:
        plt.imshow(image, aspect='auto')

    plt.show()