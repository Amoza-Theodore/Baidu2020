import os, shutil
import csv
import numpy as np

def find_bad_data():
    # config
    thresold = 0.1

    lines = []
    with open('data_order.csv', 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            lines.append(line)
    f.close()

    diff_angle = []
    for line in lines:
        if line['DATA'] == '': continue
        diff_angle.append(float(line['DIFF']))
    avg_diff = sum(diff_angle) / len(diff_angle)

    bad_data_list = []
    for line in lines:
        if line['DATA'] == '': continue
        if float(line['DIFF']) > thresold:
            bad_data_list.append(line['INDEX'])
    f.close()
    # print(len(bad_data_list))

    if os.path.exists('./bad_img'): shutil.rmtree('./bad_img')

    print('Finding bad data has started!')
    if not os.path.exists('bad_img'): os.makedirs('bad_img')
    for idx in bad_data_list:
        if os.path.exists('./img/{}.jpg'.format(idx)):
            shutil.copy('./img/{}.jpg'.format(idx), './bad_img/{}.jpg'.format(idx))
    print('Finding bad data has been completed!')

def trainlist_deal(data_path='./data.txt', score_path='./score.txt', csv_path='./data_order.csv'):
    order_list = {}
    with open(data_path, 'r') as f:
        # formula: y = (x - 500) / 2000
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = (int(line) - 500) / 2000
            order_list[idx] = line
        f.close()

    with open(score_path, 'r') as f:
        lines = f.readlines()
        score_list = {}
        for idx, line in enumerate(lines):
            score_list[idx] = float(line)
        f.close()

    with open(csv_path, 'w', newline='') as f:
        cf = csv.writer(f)
        cf.writerow(['INDEX', 'DATA', 'SCORE', 'DIFF'])
        max_key = max(max(order_list), max(score_list)) + 1
        for key in range(max_key):
            value_order = None if key not in order_list else order_list[key]
            value_score = None if key not in score_list else score_list[key]
            diff = abs(value_order - value_score) if value_order and value_score else None
            cf.writerow([key, value_order, value_score, diff])
        f.close()

def classify(data_path, img_path):
    print("Classifing has started!")

    # create new floders
    parname = os.path.dirname(data_path)
    for dirname in ('go_stright', 'turn_left', 'turn_right'):
        if not os.path.exists(os.path.join(parname, dirname)):
            os.makedirs(os.path.join(parname, dirname))

    # move files
    with open(data_path, 'r') as f:
        angles = f.readlines()
        for id in range(len(angles)):
            angle = angles[id][:-1]
            picture_path = os.path.join(img_path, str(id) + '.jpg')
            if angle == '1500':
                shutil.move(picture_path, os.path.join(parname, 'go_stright'))
            elif angle == '700':
                shutil.move(picture_path, os.path.join(parname, 'turn_right'))
            elif angle == '2100':
                shutil.move(picture_path, os.path.join(parname, 'turn_left'))
            else:
                shutil.move(picture_path, os.path.join(parname, 'go_stright'))
    f.close()
    print("Classifing has been completed!")

def rename(path, begin_id=0):
    print('Renaming has started!')

    # Organize file names.
    filenames = os.listdir(path)
    id_list = []
    for i, filename in enumerate(filenames):
        idx_num = filename.find('.')
        id_list.append(int(filename[:idx_num]))
    id_list = sorted(id_list)

    # If begin_id is less than max(id_list) then we take an intermediate quantity and do a rename operation first.
    if begin_id in id_list:
        # print("It's a normal tip: begin_id is too large, do an intermediate quantity operation first.")
        id_list_copy = id_list.copy()
        for i, id in enumerate(id_list_copy):
            os.rename(os.path.join(path, str(id)+'.jpg'), os.path.join(path, str(id+max(id_list_copy)+10)+'.jpg'))
            id_list[i] += max(id_list_copy)+10

    for i, id in enumerate(id_list):
        os.rename(os.path.join(path, str(id)+'.jpg'), os.path.join(path, str(i+begin_id)+'.jpg'))

    print('Renaming has been completed!')

def txt_2_numpy(data_path='./data.txt', npy_path='./data.npy'):
    angledata = []
    data = []
    file = open(data_path,"r")
    for line in file.readlines():
        line = line.strip('\n')
        angledata.append(int(line))
    angle = np.array(angledata)
    np.save(npy_path,angle,False)
    file.close()

if __name__ == '__main__':
    # rename(path='./data_origin/img - Copy', begin_id=1308)
    trainlist_deal(data_path='../work/data_origin/data.txt', score_path='../work/data_origin/score.txt', csv_path='../work/data_origin/data_order.csv')
    # txt_2_numpy(data_path='./data_origin/data.txt',npy_path='./data_origin/data.npy')