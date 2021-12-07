from PIL import Image, ImageEnhance, ImageOps
import os


dir = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images_copy/face"

for folder in os.listdir(dir):
    for file in os.listdir(f"{dir}/{folder}"):
        if file.startswith(folder):
            #read the image
            im = Image.open(f"{dir}/{folder}/{file}")
    
            #image brightness enhancer
            b_enhancer = ImageEnhance.Brightness(im)
            factor = 0.5 #darkens the image
            im_output = b_enhancer.enhance(factor)
            im_output.save(f'{dir}/{folder}/darkened-{file}')

            factor = 1.5 #brightens the image
            im_output = b_enhancer.enhance(factor)
            im_output.save(f'{dir}/{folder}/brightened-{file}')

            c_enhancer = ImageEnhance.Contrast(im)

            factor = 0.5 #decrease constrast
            im_output = c_enhancer.enhance(factor)
            im_output.save(f'{dir}/{folder}/less-contrast-{file}')

            factor = 1.5 #increase contrast
            im_output = c_enhancer.enhance(factor)
            im_output.save(f'{dir}/{folder}/more-contrast-{file}')

            s_enhancer = ImageEnhance.Sharpness(im)
            factor = 0.05
            im_s_1 = s_enhancer.enhance(factor)
            im_s_1.save(f'{dir}/{folder}/blurred-{file}')

            factor = 2
            im_s_1 = s_enhancer.enhance(factor)
            im_s_1.save(f'{dir}/{folder}/sharpened-{file}')

            gray_image = ImageOps.grayscale(im)
            gray_image.save(f'{dir}/{folder}/gray_{file}')