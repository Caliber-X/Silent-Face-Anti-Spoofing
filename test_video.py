# Revamped code of "test.py" for inference on live video

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredictv2
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(model_dir, device_id):
    model_test = AntiSpoofPredictv2(device_id, model_dir)
    image_cropper = CropImage()
    
    # video init params
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    # cap set dims
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap.isOpened() == False:
        print("Error in opening video stream or file")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # centre crop to aspect of widt/height to 3/4
        w = int(3/4*height)
        w1 = int(width - w) // 2
        image = frame[:, w1:w1+w]
        print(frame.shape, image.shape)
        
        result = check_image(image)
        if result is False:
            return
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, model_name)
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            print("Real Face. Score: {:.2f}.".format(value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Fake Face. Score: {:.2f}.".format(value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
        
        # Display the resulting frame
        cv2.imshow("image", image)
        # Press esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    print(args)
    test(args.model_dir, args.device_id)
