import os
import sys
import time

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


class StableVideo:
    def __init__(
        self, inputVideoPath, cut_txt, output_width, output_height, output_FPS
    ):
        logger.info("[Step 1] ->> Type 0_NEW. START.")
        self.cuttingData(cut_txt)  # create cutting dataset
        self.inputVideoPath = inputVideoPath  # folder path
        self.videolist = os.listdir(inputVideoPath)  # files list in folder
        self.videolist.sort()
        self.output_fps = output_FPS

        """ cv2 read video setting """
        cap = cv2.VideoCapture(
            os.path.join(os.path.abspath(inputVideoPath), self.videolist[0])
        )
        if cap.isOpened():
            self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.target_width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )  # get width from origin video
            self.target_height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )  # get height from origin video
            self.target_fps = cap.get(cv2.CAP_PROP_FPS)  # get FPS from origin video
        else:
            logger.warning(
                f"[Step 1(C)] Video read failed : {os.path.join(os.path.abspath(inputVideoPath), self.videolist[0])}"
            )
            return
        cap.release()

        self.output_width = output_width  # Now set dynamically
        self.output_height = output_height  # Now set dynamically

        """ stabilize setting """
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        self.number_of_iterations = 300
        self.termination_eps = 5 * 1e-4
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.number_of_iterations,
            self.termination_eps,
        )

    def stableVideoWithOutputpath(self, outputpath, display_callBack, is_Show=False):
        index = 0
        while index < len(self.videolist):
            if self.cutInfoList[index].getKey() != -1:
                cap = cv2.VideoCapture(
                    os.path.join(
                        os.path.abspath(self.inputVideoPath), self.videolist[index]
                    )
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.cutInfoList[index].getKey() - 1)
                ret, frame = cap.read()

                frameT = cv2.resize(
                    frame,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                self.target_frame = cv2.cvtColor(frameT, cv2.COLOR_BGR2GRAY)
                self.stabilization_sz = frameT.shape
                cap.release()

                break
            index = index + 1

        startTime = time.time()

        out = cv2.VideoWriter(
            outputpath,
            self.fourcc,
            self.output_fps,
            (self.output_width, self.output_height),
        )
        self.currentVideoIndex = 0  # video list index
        self.currentFrameIndex = 0  # video frame index

        self.pa = 0  # Counting every paNumber Frame for ECC
        self.paNumber = round(self.target_fps / self.output_fps)  # FPS ratio

        self.ECC_costTime = 0
        self.Read_costTime = 0
        self.Write_costTime = 0

        while self.currentVideoIndex < len(self.videolist):
            while (
                self.cutInfoList[self.currentVideoIndex].getKey() == -1
                and self.cutInfoList[self.currentVideoIndex].getStart() == -1
                and self.cutInfoList[self.currentVideoIndex].getEnd() == -1
            ):
                self.currentVideoIndex += 1
                if self.currentVideoIndex >= len(self.videolist):
                    out.release()
                    workingTime = time.time() - startTime
                    self._log_final_stats(outputpath, workingTime)
                    return
                else:
                    logger.info(
                        "[Step 1] ->> PASS : "
                        + self.videolist[self.currentVideoIndex - 1]
                    )

            self.currentFrameIndex = self.cutInfoList[self.currentVideoIndex].getStart()
            cap = cv2.VideoCapture(
                os.path.join(
                    os.path.abspath(self.inputVideoPath),
                    self.videolist[self.currentVideoIndex],
                )
            )
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                self.cutInfoList[self.currentVideoIndex].getStart(),
            )

            logger.info(
                "[Step 1] ->> Stabilize video :"
                + str(self.videolist[self.currentVideoIndex])
            )

            # Use tqdm to track frame stabilization progress
            total_frames = int(
                self.cutInfoList[self.currentVideoIndex].getEnd()
                - self.cutInfoList[self.currentVideoIndex].getStart()
            )
            for _ in tqdm(
                range(total_frames),
                desc=f"Processing {self.videolist[self.currentVideoIndex]}",
            ):
                self.IO_S = time.time()
                ret, frame = cap.read()
                self.IO_E = time.time()
                self.Read_costTime += self.IO_E - self.IO_S

                if self.pa % self.paNumber == 0:
                    frame = cv2.resize(
                        frame,
                        (self.output_width, self.output_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    self.ECC_S = time.time()
                    try:
                        (cc, self.warp_matrix) = cv2.findTransformECC(
                            self.target_frame,
                            frame_gray,
                            self.warp_matrix,
                            self.warp_mode,
                            self.criteria,
                        )
                    except:
                        (cc, self.warp_matrix) = cv2.findTransformECC(
                            self.target_frame,
                            frame_gray,
                            self.warp_matrix,
                            self.warp_mode,
                            self.criteria,
                            inputMask=None,
                            gaussFiltSize=1,
                        )
                    self.ECC_E = time.time()
                    self.ECC_costTime += self.ECC_E - self.ECC_S

                    frame_aligned = cv2.warpPerspective(
                        frame,
                        self.warp_matrix,
                        (self.stabilization_sz[1], self.stabilization_sz[0]),
                        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                    )

                    self.write_S = time.time()
                    out.write(frame_aligned)
                    self.write_E = time.time()
                    self.Write_costTime += self.write_E - self.write_S

                    if is_Show:
                        # cv2.imshow('frame', frame)
                        display_callBack(frame)
                        # 取消使用 imshow,waitKey以在linux opencv-python-headless沒有GUI的情況下運行
                        # if cv2.waitKey(1) == ord('q'):
                        #     print("Step1 break for keyboard >q< .")
                        #     break

                self.pa += 1
                self.currentFrameIndex += 1

            self.currentVideoIndex += 1

        out.release()
        workingTime = time.time() - startTime
        if is_Show:
            cv2.destroyAllWindows()
        self._log_final_stats(outputpath, workingTime)

    def _log_final_stats(self, outputpath, workingTime):
        logger.info("[Step 1] ->> Type 0 NEW.")
        logger.info(
            f"[Step 1] ->> [In % Out] FPS = paNumber : [ {self.target_fps} % {self.output_fps} ] = {self.paNumber}."
        )
        logger.info("[Step 1] ->> ECC_costTime :" + str(self.ECC_costTime))
        logger.info("[Step 1] ->> Read_costTime :" + str(self.Read_costTime))
        logger.info("[Step 1] ->> Write_costTime :" + str(self.Write_costTime))
        logger.info("[Step 1] ->> cost time :" + str(workingTime))
        logger.info("[Step 1] ->> Output file :" + outputpath)

    def cuttingData(self, cut_txt):
        f = open(cut_txt, "r")
        self.cutInfo = f.readlines()
        self.cutInfoList = []

        for i in range(len(self.cutInfo)):
            string = ""
            count = 0
            tempCut = CutInfo()
            for j in range(len(self.cutInfo[i])):
                if self.cutInfo[i][j] == "\t" or self.cutInfo[i][j] == "\n":
                    count += 1
                    tempInt = int(string)
                    string = ""

                    if count == 1:
                        tempCut.setKey(tempInt)
                    elif count == 2:
                        tempCut.setStart(tempInt)
                    elif count == 3:
                        tempCut.setEnd(tempInt)

                string = string + self.cutInfo[i][j]
            self.cutInfoList.append(tempCut)

        f.close()
        print("=======================================================")
        for i in range(len(self.cutInfoList)):
            print(
                f"Key : {self.cutInfoList[i].getKey()} \t Start : {self.cutInfoList[i].getStart()} \t End : {self.cutInfoList[i].getEnd()}"
            )
        print("=======================================================")


class CutInfo:
    def __init__(self):
        self.key = -1
        self.start = -1
        self.end = -1

    def setKey(self, key):
        self.key = int(key)

    def setStart(self, start):
        self.start = int(start)

    def setEnd(self, end):
        self.end = int(end)

    def getKey(self):
        return self.key

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end


if __name__ == "__main__":
    print("::::Stabilization.py Example::::")
    print("> Please put the video which need to stabilize in the same directory,")
    print("> and make the file name as '1.mp4'.")
    print("> Then there will be a file output as '2.avi'.")

    current_STAB = StableVideo("1.mp4", "cut.txt", 1920, 1080)
    current_STAB.stableVideoWithOutputpath(outputpath="2.avi")


def stab_main(
    stab_input,
    stab_output,
    show,
    cut_txt,
    output_height,
    output_width,
    output_FPS,
    display_callBack,
):
    current_STAB = StableVideo(
        stab_input, cut_txt, output_width, output_height, output_FPS
    )
    current_STAB.stableVideoWithOutputpath(stab_output, display_callBack, show)
