import cv2
import torch
from enum import Enum
import numpy as np
import time


class ModelType(Enum):
    DPT_HYBRID = "DPT_Hybrid"
    DPT_LARGE = "DPT_Large"
    MIDAS_SMALL = "MiDaS_small"


class MiDas():
    def __init__(self, model_type=ModelType.DPT_LARGE):
        self.MiDas = torch.hub.load("intel-isl/MiDas", model_type.value)
        self.model_type = model_type

    def useCUDA(self):
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        self.MiDas.to(self.device)
        self.MiDas.eval()

    def transform(self):
        print("transform")
        midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")
        if self.model_type in (ModelType.DPT_LARGE, ModelType.DPT_HYBRID):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.MiDas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        return depth_map

    def livePredict(self):
        print("Starting webcam (press q to exit)...")
        capObj = cv2.VideoCapture(0)
        if not capObj.isOpened():
            print("Error: Could not open video capture.")
            return

        prev_time = time.time()
        while True:
            ret, frame = capObj.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            depthMap = self.predict(frame)
            combined = np.hstack((frame, depthMap))

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Overlay FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(depthMap, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            combined = np.hstack((frame, depthMap))

            cv2.imshow('Combined', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capObj.release()
        cv2.destroyAllWindows()


def run(model_type: ModelType):
    midas_obj = MiDas(model_type)
    midas_obj.useCUDA()
    midas_obj.transform()
    midas_obj.livePredict()


if __name__ == '__main__':
    run(ModelType.DPT_HYBRID)
