def dataset(image_path):
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])

    frame = cv2.imread(image_path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)

    img = Image.fromarray(mask)
    img.show()
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0;
    img = np.expand_dims(img, axis=0)
    img_256 = Image.fromarray(frame)
    return img_256, img;

if __name__ == '__main__':
    img,mask = dataset(image_path)
    cv2.imshow(img, mask)