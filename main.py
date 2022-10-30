#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, request, redirect, flash, jsonify,send_from_directory, send_file
from werkzeug.utils import secure_filename
from toxicity import ToxicCommentsDetector
from PIL import ImageDraw, Image
from autocorrect import Speller
import logging

import os
from functools import cmp_to_key

import cv2
import numpy as np
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

import ml.imgproc as imgproc
import ml.test as test
from ml import file_utils
from ml.craft import CRAFT
from ml.recognition.dataset import AlignCollate, RawDataset
from ml.recognition.model import Model
from ml.recognition.utils import AttnLabelConverter

trained_model_craft = 'ml/craft_mlt_25k.pth'
trained_model_recognition = 'ml/best_accuracy.pth'
output = './out/'
cuda = False

app = Flask(__name__)

if __name__ == '__main__':
    net = CRAFT()
    print('Loading weights from checkpoint (' + trained_model_craft + ')')
    if cuda:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_craft)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_craft, map_location='cpu')))

    net.eval()
    print('model evaluated')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    converter = AttnLabelConverter(
        '0123456789,.?!:&*()%-=+ abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    num_class = len(converter.character)
    input_channel = 3
    opt = {
        'Transformation': 'None',
        'Prediction': 'Attn',
        'SequenceModeling': 'BiLSTM',
        'FeatureExtraction': 'ResNet',
        'input_channel': input_channel,
        'output_channel': 512,
        'hidden_size': 256,
        'num_class': num_class,
        'imgH': 32,
        'imgW': 100,
        'batch_size': 192,
        'workers': 4,
        'rgb': True,
        'batch_max_length': 25,
    }
    print('loading recognition model')
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(trained_model_recognition, map_location=device))
    AlignCollate_demo = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=False)
    # MODEL LOADING #

    words_correct = set(open('dictionary/russian_surnames.txt', encoding='utf-8').read().splitlines())
    words_censor = open('dictionary/censor_data.txt', encoding='utf-8').read().splitlines()
    allowed_extensions = {'png',
                          'jpg',
                          'jpeg'}
    similarity = {'а': ['а', 'a', '@'],
                  'б': ['б', '6', 'b'],
                  'в': ['в', 'b', 'v'],
                  'г': ['г', 'r', 'g'],
                  'д': ['д', 'd'],
                  'е': ['е', 'e'],
                  'ё': ['ё', 'e'],
                  'ж': ['ж', 'zh', '*'],
                  'з': ['з', '3', 'z'],
                  'и': ['и', 'u', 'i'],
                  'й': ['й', 'u', 'i'],
                  'к': ['к', 'k', 'i{', '|{'],
                  'л': ['л', 'l', 'ji'],
                  'м': ['м', 'm'],
                  'н': ['н', 'h', 'n'],
                  'о': ['о', 'o', '0'],
                  'п': ['п', 'n', 'p'],
                  'р': ['р', 'r', 'p'],
                  'с': ['с', 'c', 's'],
                  'т': ['т', 'm', 't'],
                  'у': ['у', 'y', 'u'],
                  'ф': ['ф', 'f'],
                  'х': ['х', 'x', 'h', '}{'],
                  'ц': ['ц', 'c', 'u,'],
                  'ч': ['ч', 'ch'],
                  'ш': ['ш', 'sh'],
                  'щ': ['щ', 'sch'],
                  'ь': ['ь', 'b'],
                  'ы': ['ы', 'bi'],
                  'ъ': ['ъ'],
                  'э': ['э', 'e'],
                  'ю': ['ю', 'io'],
                  'я': ['я', 'ya']}

    main_folder = os.getcwd()
    toxicDetector = ToxicCommentsDetector()
    spell = Speller(lang='ru')

    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.config['UPLOAD_FOLDER'] = f'{main_folder}\\upload_folder'
    app.config["TEXT_FOLDER"] = f'{main_folder}\\txt_folder'
    app.config['JSON_AS_ASCII'] = False


    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


    def check_phrase(input_phrase):
        def distance(a, b):
            n, m = len(a), len(b)
            if n > m:
                # Make sure n <= m, to use O(min(n, m)) space
                a, b = b, a
                n, m = m, n

            current_row = range(n + 1)  # Keep current and previous row, not entire matrix
            for i in range(1, m + 1):
                previous_row, current_row = current_row, [i] + [0] * n
                for j in range(1, n + 1):
                    add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                    if a[j - 1] != b[i - 1]:
                        change += 1
                    current_row[j] = min(add, delete, change)

            return current_row[n]

        phrase = input_phrase.lower().replace(" ", "")

        for key, value in similarity.items():
            for letter in value:
                for phr in phrase:
                    if letter == phr:
                        phrase = phrase.replace(phr, key)

        for word in words_censor:
            for part in range(len(phrase)):
                fragment = phrase[part: part + len(word)]
                if distance(fragment, word) <= len(word) * 0.15:
                    return True


    def draw_boxes(img, line, color='red', width=4):
        draw = ImageDraw.Draw(img)
        p0, p1, p2, p3 = line[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        return img


    def coord_alignment(cv_img, cord):
        rect = np.zeros((4, 2), dtype="float32")

        s = np.sum(cord, axis=1)
        rect[0] = cord[np.argmin(s)]
        rect[2] = cord[np.argmax(s)]

        diff = np.diff(cord, axis=1)
        rect[1] = cord[np.argmin(diff)]
        rect[3] = cord[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        width_a = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
        width_b = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        height_b = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)

        scan = cv2.warpPerspective(cv_img, transform_matrix, (max_width, max_height))

        return scan


    def crop(pts, image):
        """
        Takes inputs as 8 points
        and Returns cropped, masked image with a white background
        """
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = image[y:y + h, x:x + w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        bg = np.ones_like(cropped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        return dst2, (y, h, x)


    def generate_words(image_name, score_bbox, image):
        num_bboxes = len(score_bbox)
        for num in range(num_bboxes):
            bbox_coords = list(score_bbox.values())[num]
            if bbox_coords.size != 0:
                pts = np.array(bbox_coords).astype('int32')
                pts = np.where(pts < 0, 0, pts)
                if np.all(pts) >= 0:
                    word, (y, h, x) = crop(pts, image)

                    folder = '/'.join(image_name.split('/')[:-1])

                    if not os.path.isdir(os.path.join(output + folder)):
                        os.makedirs(os.path.join(output + folder))

                    try:
                        file_name = os.path.join(output + image_name)
                        cv2.imwrite(
                            file_name + '_{}_{}_{}.jpg'.format(y, h, x), word)
                        # print('Image saved to ' + file_name + '_{}_{}.jpg'.format(x, y))
                    except:
                        continue


    def filter_threshold(data, threshold):
        result = []
        condition = (lambda x, y: abs(x - y) > threshold)
        for element in data:
            if all(condition(element, other) for other in result):
                result.append(element)
        return result


    def ml_get_text(path_img):
        image = imgproc.loadImage(path_img)

        bboxes, polys, score_text, det_scores = \
            test.test_net(net, image, 0.4, 0.3, 0.4, cuda, False, {'canvas_size': 1280, 'mag_ratio': 1.5})

        bbox_score = {}

        for box_num in range(len(bboxes)):
            key = str(det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key] = item

        # file_utils.saveResult(image_path, cropped[:, :, ::-1], polys, dirname=output)
        # print(bbox_score)
        img = np.array(image[:, :, ::-1])
        with open(path_img, 'w') as f:
            for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)

        # Save result image
        cv2.imwrite(path_img, img)

        generate_words('test', bbox_score, image)

        demo_data = RawDataset(root=output, opt=opt)  # use RawDataset
        demo_loader = DataLoader(
            demo_data, batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=int(opt['workers']),
            collate_fn=AlignCollate_demo, pin_memory=True)

        model.eval()

        results = []
        for image_tensors, image_path_list in demo_loader:
            with torch.no_grad():
                batch_size = image_tensors.size(0)
                image_dev = image_tensors.to(device)
                # For max length prediction

                length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

                if 'CTC' in opt['Prediction']:
                    preds = model(image_dev, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(image_dev, text_for_pred, is_train=False)
                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = func.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt['Prediction']:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                # confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                y_, h_, x_ = img_name[11:-4].split('_')
                # print(f'{x+" "+y:25s}\t {pred:25s}\t {confidence_score:0.4f}')
                results.append([pred, [int(y_), int(h_), int(x_)]])

        verticals = np.array(list(map(lambda x: x[1], results)))
        threshold = np.average(verticals[:, 1]) * 0.8

        rows = np.array(filter_threshold(verticals[:, 0], threshold))
        word = [None] * rows.size
        for i in results:
            idx = (np.abs(rows - i[1][0])).argmin()
            if word[idx] is None:
                word[idx] = []
            row_ = word[idx]
            row_.append((i[0], i[1][2]))
        for i, x in enumerate(word):
            word[i] = ' '.join(map(lambda x: x[0], sorted(x, key=cmp_to_key(lambda item1, item2: item1[1] - item2[1]))))

        return '\n'.join(word)


    @app.route("/import/<string:ethics>&<string:mistakes>&<string:censor>&<string:lang>", methods=["POST", "GET"])
    def import_page(ethics, mistakes, censor, lang):
        if request.method == 'POST':
            if 'file' in request.files:
                file_img = request.files['file']
                img_filename = secure_filename(file_img.filename)
                txt_filename = f'{img_filename.split(".")[0]}.txt'
                path_img = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                path_txt = os.path.join(app.config["TEXT_FOLDER"], txt_filename)

                if not img_filename == '':
                    if file_img and allowed_file(img_filename):
                        file_txt = open(path_txt, mode='w', encoding='UTF-8')
                        file_img.save(path_img)

                        alarm_ethics = False
                        alarm_censor = False

                        get_text = ml_get_text(path_img)

                        if bool(censor) or bool(mistakes):
                            words_list = get_text.split(' ')
                            for elem_index in range(len(words_list)):
                                elem = words_list[elem_index]

                                if bool(censor) and check_phrase(elem):
                                    alarm_censor = True

                                if bool(mistakes) and not elem in words_correct:
                                    words_list[elem_index] = spell(elem)
                            get_text = ' '.join(words_list)

                        if ethics:
                            if toxicDetector.predict([get_text])[0] > 0.85:
                                alarm_ethics = True

                        file_txt.write(get_text)
                        file_txt.close()

                        return jsonify({'alarm_ethics': alarm_ethics,
                                        'alarm_censor': alarm_censor,
                                        'img_filename': img_filename,
                                        'txt_filename': txt_filename,
                                        'result': get_text})
                else:
                    flash('No selected file')

                    return redirect(request.url)
            else:
                flash('No file part')

                return redirect(request.url)

        return '''
            <!doctype html>
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
            '''


    @app.route("/get-image/<path:file_name>", methods=['GET', 'POST'])
    def get_image(file_name):
        return send_file(os.path.join(app.config["UPLOAD_FOLDER"], file_name))


    @app.route("/get-texts/<path:file_name>", methods=['GET', 'POST'])
    def get_texts(file_name):
        return send_file(os.path.join(app.config['TEXT_FOLDER'], file_name), as_attachment=True)

    app.run(host='0.0.0.0', port=4000, debug=False)
