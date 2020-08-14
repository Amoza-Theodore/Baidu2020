import cv2
import numpy as np
import paddle.fluid as fluid
from PIL import Image
import getopt
import os

model_save_dir = "model_infer"
infer_path = "img"
score_save_path = ''

params = "model_infer/params"
model = "model_infer/model"

def dataset(image_path):
    lower_hsv = np.array([25, 75, 195])
    upper_hsv = np.array([55, 255, 255])

    frame = cv2.imread(image_path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    mask = mask0  # + mask1

    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

with fluid.scope_guard(inference_scope):
    print("Infering has started!")

    [inference_program,
     feed_target_names,
     target_var] = fluid.io.load_inference_model(model_save_dir,
                                                 infer_exe,
                                                 model_filename='model',
                                                 params_filename='params')

    # img_names -> img_id
    img_ids = []
    img_names = os.listdir(infer_path)
    for img_name in img_names:
        if img_name[-3:] != 'jpg': continue
        img_ids.append(int(img_name[:-4]))

    # sort img_id and turn it to img_names
    tmp = []
    img_ids = sorted(img_ids)
    for img_id in img_ids:
        tmp.append(str(img_id) + '.jpg')
    img_names = tmp

    # infer and storage it to results
    results = []
    for img_name in img_names:
        img_path = os.path.join(infer_path, img_name)
        img = dataset(img_path)

        infer_results = infer_exe.run(program=inference_program,
                                      feed={feed_target_names[0]: img},
                                      fetch_list=target_var)

        results.append(infer_results[0][0][0])

    with open(os.path.join(score_save_path, 'score.txt'), 'w') as f:
        for result in results:
            f.write(str(result) + '\n')
    f.close()
    print("Infering has been completed!")