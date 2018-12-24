from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np
mode="val"
image_dir = "/media/hszc/model/detao/data/bdd/bdd/bdd100k/labels_seg/{}".format(mode)
val_json = "/media/hszc/model/detao/data/bdd/bdd/bdd100k/labels/bdd100k_labels_images_{}.json".format(mode)
val_pd = pd.read_json(open(val_json))
print(val_pd.head())


def poly2patch(vertices, types, closed=False, alpha=1., color=None):
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [v for v in vertices]
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)
    #
    # if color is None:
    #     color = random_color()

    # print(codes, points)
    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,  # if not closed else 'none',
        lw=1 if closed else 2 * 1, alpha=alpha,
        antialiased=False, snap=True)


def get_areas_v0(objects):
    # print(objects['category'])
    return [o for o in objects
            if 'poly2d' in o and o['category'] == 'drivable area']


def draw_drivable(objects):
    plt.draw()

    objects = get_areas_v0(objects)
    print(len(objects))
    colors = np.array([[0, 0, 0, 255],
                       [217, 83, 79, 255],
                       [91, 192, 222, 255]]) / 255
    colors = np.array([[0, 0, 0],
                       [255, 0, 0],
                       [0, 255, 0]]) / 255
    for obj in objects:
        if obj['attributes']['areaType'] == 'direct':
            color = colors[1]
        else:
            color = colors[2]
        alpha = 0.5
        for poly in obj['poly2d']:
            ax.add_patch(poly2patch(
                poly['vertices'], poly['types'], closed=poly['closed'],
                alpha=alpha, color=color))

    ax.axis('off')

    # ax.add_patch(poly2patch(obj['poly2d'],types closed=False, alpha=alpha, color=color))


for data, img_name in zip(val_pd["labels"], val_pd["name"]):
    # print(data)
    dpi = 80
    w = 16
    h = 9
    image_width = 1280
    image_height = 720
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    # plt.connect('key_release_event', next_image)
    # image_path = os.path.join(image_dir, img_name)
    out_path = os.path.join(image_dir, img_name).replace(".jpg", ".png")
    # img = mpimg.imread(image_path)
    # im = np.array(img, dtype=np.uint8)
    # ax.imshow(im, interpolation='nearest', aspect='auto')
    ax.set_xlim(0, image_width - 1)
    ax.set_ylim(0, image_height - 1)
    ax.invert_yaxis()
    ax.add_patch(poly2patch(
        [[0, 0], [0, image_height - 1],
         [image_width - 1, image_height - 1],
         [image_width - 1, 0]], types='LLLL',
        closed=True, alpha=1., color='black'))
    draw_drivable(data)
    # plt.show()
    fig.savefig(out_path, dpi=dpi)
    plt.close()