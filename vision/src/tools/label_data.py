'''
Displays images in a directory and allows you to move them to other directories based on the key bindings you set. Used for manually labeling data. Left arrow key goes back if you want to reclassify something.
'''

import os, sys
import tkinter as Tkinter
from PIL import Image, ImageTk
from os.path import join
import glob

bad_ims = []
good_ims = []
medium_ims = []
other = []
current = 0   # index of image we are viewing

image_dir = sys.argv[1]

# pred_dir = sys.argv[1]
# limit = int(sys.argv[2])
# probability_cutoff = float(sys.argv[3])
# file_grep = sys.argv[4]
# pred_files = sorted(glob.glob(join(pred_dir, file_grep)))


# def get_menu_urls():
#     all_menu_urls = []
#     for pred_file in pred_files:
#         print(pred_file)
#         with open(pred_file, 'r') as inf:
#             try:
#                 next(inf)
#             except:
#                 continue
#
#             menu_urls = [x.split(',')[2] for x in inf.readlines() if x.split(',')[3] == 'menu' and float(x.split(',')[4]) >= probability_cutoff]
#
#             all_menu_urls.extend(menu_urls)
#
#     with open('menu_urls','w') as outf:
#         for menu_url in all_menu_urls:
#             outf.write('%s\n' % menu_url)



seen_images = []

def is_image(image_file):
    try:
        Image.open(image_file)
    except:
        return False
    return True

image_list = [x for x in os.listdir(image_dir)]

print('{} total images'.format(len(image_list)))

filename = image_list[0]

def q_key(event):
    print('quitting')
    event.widget.quit()
    print('Bad: {}'.format(bad_ims))
    print('Good: {}'.format(good_ims))
    print('Medium: {}'.format(medium_ims))
    print('Other: {}'.format(other))

    write_image_paths()

def b_key(event):
    print('bad: {}'.format(filename))
    os.rename(filename, join('../true_bad', filename))
    bad_ims.append(filename)
    seen_images.append(filename)
    move(+1, event)

def g_key(event):
    print('good: {}'.format(filename))
    os.rename(filename, join('../true_good', filename))
    good_ims.append(filename)
    seen_images.append(filename)
    move(+1, event)

def m_key(event):
    print('menu: {}'.format(filename))
    os.rename(filename, join('../true_medium', filename))
    medium_ims.append(filename)
    seen_images.append(filename)
    move(+1, event)
#
# def r_key(event):
#     print('receipt: {}'.format(filename))
#     os.rename(filename, join('../true_receipt', filename))
#     receipts.append(filename)
#     seen_images.append(filename)
#     move(+1, event)

def o_key(event):
    print('other: {}'.format(filename))
    os.rename(filename, join('../true_other', filename))
    other.append(filename)
    seen_images.append(filename)
    move(+1, event)

def left_arrow(event):
    last_image = seen_images.pop()
    print('left on: {}'.format(filename))
    print('back to: {}'.format(last_image))
    if bad_ims[-1] == last_image:
        os.rename(join('../true_bad', last_image), last_image)
        bad_ims.pop()
    elif good_ims[-1] == last_image:
        os.rename(join('../true_good', last_image), last_image)
        good_ims.pop()
    elif other[-1] == last_image:
        os.rename(join('../true_other', last_image), last_image)
        other.pop()
    elif medium_ims[-1] == last_image:
        os.rename(join('../true_medium', last_image), last_image)
        other.pop()
    else:
        pass

    move(-1, event)

def write_image_paths():
    with open('good', 'w+') as outf:
        for good in good_ims:
            outf.write('%s\n' % good)
    with open('bad', 'w+') as outf:
        for bad in bad_ims:
            outf.write('%s\n' % bad)
    with open('medium', 'w+') as outf:
        for medium in medium_ims:
            outf.write('%s\n' % medium)

def move(delta, event):
    global current, image_list, filename, label_image, iamge1
    if not (0 <= current + delta < len(image_list)):
        event.widget.quit()
        # tkMessageBox.showinfo('End', 'No more image.')
        print('no more images')
        print('Bad: {}'.format(bad_ims))
        print('Good: {}'.format(good_ims))
        print('Medium: {}'.format(medium_ims))
        print('Other: {}'.format(other))

        write_image_paths()
        return
    current += delta
    filename = image_list[current]
    image = Image.open(filename)
    photo = ImageTk.PhotoImage(image)
    label['image'] = photo
    label.photo = photo


root = Tkinter.Tk()

label = Tkinter.Label(root, compound=Tkinter.TOP)
label.pack()

frame = Tkinter.Frame(root)
frame.pack()

# root.bind('m', lambda event: m_key(event))
# root.bind('r', lambda event: r_key(event))
root.bind('o', lambda event: o_key(event))
root.bind('q', lambda event: q_key(event))
root.bind('b', lambda event: b_key(event))
root.bind('m', lambda event: m_key(event))
root.bind('g', lambda event: g_key(event))

root.bind('<Left>', lambda event: left_arrow(event))

move(0,frame)

root.mainloop()


