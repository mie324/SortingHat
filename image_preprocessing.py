from PIL import Image
import glob


def save_img(fd, pm):
    image_list = []
    for fmt in ['jpg', 'jpeg', 'png']:
        for filename in glob.glob('./images_Google/{}/*.{}'.format(fd, fmt)):
            im = Image.open(filename)
            image_list.append(im)
    for i, img in enumerate(image_list):
        x, y = img.size
        d = abs(x - y) // 2
        if x < y:
            img = img.crop((0, d, x, y - d))
        else:
            img = img.crop((d, 0, x - d, y))
        img.resize((128, 128), Image.BILINEAR).save('./images_Google/{}/{}{}.png'.format(pm, pm, i))


if __name__ == "__main__":
    folder = 'fast food (wax OR foil) paper'
    params = 'fastfoodandwrapper'
    # for folder, params in [('n00021265_food', 'food'), ('n03958227_plastic_bag', 'plasticbag'),
    #                        ('n03983396_glass_bottle', 'glassbottles'), ('n06267145_newspaper', 'newspaper'),
    #                        ('n07927512_popcan', 'popcan')]:
    save_img(folder, params)

