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
        img.resize((128, 128), Image.BILINEAR).save('./images_Google/{}/{}{}.png'.format(pm[0], pm[1], i))


if __name__ == "__main__":
    folder = 'paper coffee cups -art'
    params = ('coffeecups', 'cp')
    save_img(folder, params)

