import numpy as np

def gan_make_data(windows, latent, y):
    x = []
    y_7d = []
    y_1d = []
    for i in range(windows, latent.shape[0]):
        tmp_x = latent[i - windows:i, :]
        tmp_y_7d = y[i - 6:i + 1]
        tmp_y_1d = y[i]
        x.append(tmp_x)
        y_7d.append(tmp_y_7d)
        y_1d.append(tmp_y_1d)
    x = np.array(x)
    y_7d = np.array(y_7d)
    y_1d = np.array(y_1d)

    return x, y_7d, y_1d