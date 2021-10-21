import cv2
import time
import numpy as np
from threading import Thread
from model.simple_CNN import SimpleCNN


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, img_size=1280):
        self.img_size = img_size
        self.img = None
        sources = ['0']
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, 0, s), end='')
            cap = cv2.VideoCapture(0)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.img = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(self.img, new_shape=self.img_size)[0].shape], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.img = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.img = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.img.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(img0, new_shape=self.img_size, auto=self.rect)[0]]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, [img0], None

    def __len__(self):
        return 0


if __name__ == "__main__":

    model = SimpleCNN(dropout=0)
    model.build(input_shape=(None, 640, 480, 3))
    model.load_weights("weights.h5")

    t0 = time.time()
    for path, img, im0s, vid_cap in LoadStreams():

        t1 = time.time()

        img = np.array(img, dtype=np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        t2 = time.time()
        prediction = model(img)[0]
        # prediction = ["Full", "Less", "Empty"][np.argmax(prediction)]
        t3 = time.time()

        # Stream results
        cv2.imshow(path[0], im0s[0])
        if cv2.waitKey(1) == ord("q"):  # q to quit
            raise StopIteration

        total_cost_str  = f"Total cost time: {t3-t1:.3f}s"
        fps_str         = f"FPS: {1/(t3-t1):.3f}"
        img_process_str = f"Image process time: {t2-t1:.3f}s"
        pred_str        = f"Prediction time: {t3-t2:.3f}s"
        print(f"Prediction: {prediction}. ({total_cost_str} / {fps_str} / {img_process_str} / {pred_str})")