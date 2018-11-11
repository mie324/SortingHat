from google_images_download import google_images_download

# class instantiation
response = google_images_download.googleimagesdownload()

# creating list of arguments
# list of used keywords
# "paper coffee cups -art,juice box -art,snack packaging garbage, take out box"

arguments = {"keywords": "fast food (wax OR foil) paper",
             "limit": 500, "print_urls": True,
             "chromedriver": "C:\\Users\\iatey\\PycharmProjects\\mie324\\chromedriver.exe",
             "output_directory": './SortingHat/images_Google'}
# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)
