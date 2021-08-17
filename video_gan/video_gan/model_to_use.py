import glob
print(glob.glob('./img/[0-9].jpg')+glob.glob('./img/[0-9][0-9].jpg')+glob.glob('./img/[0-9][0-9][0-9].jpg'))