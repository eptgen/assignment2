import imageio

for it in range(7):
    imgs = []
    for frame in range(10):
        fname = str(it * 100) + "_" + str(frame) + "_gt.png"
        img = imageio.imread(fname)
        imgs.append(img)
    imageio.mimwrite(str(it * 100) + "_gt.gif", imgs, fps = 15, loop = 0)