import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import argparse
# -------------------------------------------------------------
# 以下請確保有安裝 / 放在相對位置的自定義套件
# (與你原程式相同的 import)
# -------------------------------------------------------------
from .models_pkg.experimental import attempt_load
from .utils_pkg.datasets import LoadStreams, LoadImages, letterbox
from .utils_pkg.general import (
    check_img_size, non_max_suppression, rotate_non_max_suppression,
    apply_classifier, scale_coords, r_scale_coords_new, xyxy2xywh,
    xywhtheta24xy_new, draw_one_polygon, plot_one_box, strip_optimizer
)
from .utils_pkg.torch_utils import select_device, load_classifier, time_synchronized

from .models_pkg.models import *
from .models_pkg.experimental import *
from .utils_pkg.datasets import *
from .utils_pkg.general import *
from .utils_pkg.remove_duplicate_box_buildin import *
import sys, time
from textwrap import shorten

def clear_last_n_lines(n: int):
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # 游標上移一行
        sys.stdout.write("\x1b[2K")  # 清除該行
    sys.stdout.flush()
# -------------------------------------------------------------
# 類別映射 (如原程式)
# -------------------------------------------------------------
merged_cls_dict = {
    'person': '0', 'bike': '1', 'moto': '2', 'sedan': '3',
    'truck': '4', 'bus': '5', 'tractor': '6', 'trailer': '7'
}

# 顯示視窗大小 (可自行調整)
final_display_width = 1920
final_display_height = 1080


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r', encoding='utf-8') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def obb_object_detect(
    source_path='',
    output_detect_result_path='',
    yolo_model='',
    enable_cropping=True,        # <--- 是否啟用裁切模式
    overlap=0.1,                 # <--- 裁切模式時區塊重疊比例 (0~1)
    show_crop_visual=False,       # <--- 是否顯示裁切區塊偵測
    show_final_visual=False,     # <--- 是否顯示最終結果 (原程式 view_img)
    scale_up_factor=1.0          # <--- 先放大圖片倍率 (預設1.0不放大)
):
    """
    enable_cropping = True/False，決定是否使用裁切模式。
    overlap：裁切區塊重疊比例(0~1之間)，例如 0.1 = 相鄰區塊重疊 10%。
    show_crop_visual：顯示每塊裁切區域的推論結果 (Debug 用)。
    show_final_visual：顯示整張影像的最終推論結果。
    scale_up_factor：>1.0 時會先將影像放大後再進行推論，最後再將結果縮回原大小。
    """

    print("[DEBUG] obb_object_detect() - 開始")
    print(f"[DEBUG] 參數：enable_cropping={enable_cropping}, overlap={overlap}, "
          f"show_crop_visual={show_crop_visual}, show_final_visual={show_final_visual}, "
          f"scale_up_factor={scale_up_factor}")

    # ---------------------------------------------------------
    # 設定 YOLO 相關路徑 (依照原本程式結構)
    # ---------------------------------------------------------
    yolo_root = Path(__file__).parent
    cfg_file_path = yolo_root / 'cfg' / 'yolov4-pacsp-mish-9anchor-headcxy-vehicle8cls-1920_w_layerID_wo_whrepair.cfg'
    weight_file_path = yolo_root / 'weights' / yolo_model
    cls_file_path = yolo_root / 'data' / 'classes_vehicle8cls.txt'

    with torch.no_grad():
        # 這裡為了與原程式相容，保留原先一些變數 (可視需求調整)
        out, source, weights, view_img, save_txt, imgsz, cfg, names_file, theta_format, \
        save_img_before_nms, which_dataset, save_label_car_tool_format_txt, \
        save_vehicle8cls_IOTMOTC_format_txt, save_dota_txt = (
            'vehicle8cls_detect',     # out
            source_path,              # source
            [str(weight_file_path)],  # weights
            show_final_visual,        # view_img -> 是否顯示最終畫面
            False,                    # save_txt (預設 False)
            1920,                     # imgsz
            str(cfg_file_path),       # cfg
            str(cls_file_path),       # names
            'dhxdhy',                 # theta_format
            False,                    # save_img_before_nms
            'vehicle8cls',            # which_dataset
            False,                    # save_label_car_tool_format_txt
            True,                     # save_vehicle8cls_IOTMOTC_format_txt
            False                     # save_dota_txt
        )

        gpu_device = '0'
        agnostic_nms = True
        conf_thres = 0.3
        iou_thres = 0.1

        webcam = (source == '0'
                  or source.startswith('rtsp')
                  or source.startswith('http')
                  or source.endswith('.txt'))
        num_extra_outputs = get_number_of_extra_outputs(theta_format)

        # ---------------------------------------------------------
        # 初始化 device
        # ---------------------------------------------------------
        device = select_device(gpu_device)
        half = (device.type != 'cpu')  # 只有 CUDA 時才可 half

        # ---------------------------------------------------------
        # 建立模型
        # ---------------------------------------------------------
        model = Darknet(cfg, imgsz, extra_outs=theta_format, num_extra_outputs=num_extra_outputs).cuda()
        try:
            state_dict = torch.load(weights[0], map_location=device)
            model.load_state_dict(state_dict['model'])
            print(f"[DEBUG] 成功載入權重：{weights[0]}")
        except:
            print("[DEBUG] 嘗試 load_darknet_weights() (非 .pt 檔) ...")
            load_darknet_weights(model, weights[0])
        model.to(device).eval()

        if half:
            model.half()
            print("[DEBUG] 使用 FP16 模式")

        # 第二階段 classifier (此處預設不啟用)
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
            modelc.to(device).eval()

        # ---------------------------------------------------------
        # 建立 Dataloader
        # ---------------------------------------------------------
        if webcam:
            # 網路串流
            print("[DEBUG] 使用 LoadStreams (webcam/RTSP)")
            view_img = True
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            print("[DEBUG] 使用 LoadImages (圖片 / 影片)")
            dataset = LoadImages(source, img_size=imgsz)

        # 讀取類別檔
        names = load_classes(names_file)

        # 暖機推論
        img_dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(img_dummy.half() if half else img_dummy)
        print("[DEBUG] 完成暖機推論")

        # ---------------------------------------------------------
        # 主迴圈：讀取每張圖 / frame
        # ---------------------------------------------------------
        t0 = time.time()
        frame_count = 0
        prev_line = 0
        
        for path, img, im0s, vid_cap in dataset:
            frame_count += 1
            msg = []
            msg.append(f"|| [DEBUG] 處理第 {frame_count} 個影格/影像，path={path}")

            if webcam:
                # 多路串流 -> im0s 可能是 list
                raise NotImplementedError("[DEBUG] 尚未實作多路 webcam 的裁切示例")
            else:
                im0_original = im0s  # 單張(或影片frame)

            # (1) 先放大
            if scale_up_factor != 1.0:
                h0, w0 = im0_original.shape[:2]
                new_w = int(w0 * scale_up_factor)
                new_h = int(h0 * scale_up_factor)
                msg.append(f"|| [DEBUG] 放大影像: 原({w0}x{h0}) -> ({new_w}x{new_h})")
                im0_for_inference = cv2.resize(im0_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                im0_for_inference = im0_original

            # (2) 判斷是否啟用裁切
            if not enable_cropping:
                # print("[DEBUG] 未啟用裁切 -> 單張整圖推論")
                # -------------------------------------
                # 若不裁切，可依原流程直接 letterbox & 推論
                # -------------------------------------

                sub_img = letterbox(im0_for_inference, new_shape=imgsz, auto=False)[0]
                sub_img = sub_img[:, :, ::-1].transpose(2, 0, 1)
                sub_img = np.ascontiguousarray(sub_img)

                img_t = torch.from_numpy(sub_img).to(device)
                img_t = img_t.half() if half else img_t.float()
                img_t /= 255.0
                if img_t.ndimension() == 3:
                    img_t = img_t.unsqueeze(0)

                t1 = time_synchronized()
                pred = model(img_t, augment=False)[0]
                pred = rotate_non_max_suppression(
                    pred,
                    conf_thres,
                    iou_thres,
                    classes=[],
                    agnostic=agnostic_nms,
                    theta_format=theta_format,
                    num_extra_outputs=num_extra_outputs,
                    whichdataset=which_dataset
                )
                t2 = time_synchronized()
                msg.append(f"|| [DEBUG] 推論耗時: {t2 - t1:.3f}s")

                # 取出最終結果
                if len(pred) > 0 and pred[0] is not None and len(pred[0]) > 0:
                    det_final = pred[0]
                else:
                    det_final = torch.zeros((0, 10), device=device)

                # 如果要做 second classifier
                if classify and len(det_final) > 0:
                    det_final = apply_classifier([det_final], modelc, img_t, im0_for_inference)[0]

                # (3) scale 回 im0_for_inference
                if len(det_final) > 0:
                    det4pts = xywhtheta24xy_new(det_final[:, :5])
                    det4pts = r_scale_coords_new((imgsz, imgsz), det4pts, im0_for_inference.shape)
                    det_final = torch.cat((det4pts, det_final[:, 5:]), dim=1)

                # (4) 若有放大 -> 再縮回 im0_original
                if scale_up_factor != 1.0 and len(det_final) > 0:
                    det4pts = det_final[:, :8]
                    other_part = det_final[:, 8:]
                    det4pts_np = det4pts.cpu().numpy()
                    for i_box in range(det4pts_np.shape[0]):
                        for j in range(0, 8, 2):
                            det4pts_np[i_box, j]   /= scale_up_factor
                            det4pts_np[i_box, j+1] /= scale_up_factor
                    det4pts_t = torch.from_numpy(det4pts_np).to(device)
                    det_final = torch.cat([det4pts_t, other_part], dim=1)

                det = det_final

            else:
                # -------------------------------------------------
                # 裁切模式 (修正版迴圈)
                # -------------------------------------------------
                msg.append(f"|| [DEBUG] 啟用裁切模式, overlap={overlap}")
                im0_infer = im0_for_inference
                h_infer, w_infer = im0_infer.shape[:2]

                tile_size = imgsz
                step = int(tile_size * (1 - overlap))
                msg.append(f"|| [DEBUG] tile_size={tile_size}, step={step}")

                final_boxes_in_infer = []
                y0 = 0

                while y0 < h_infer:
                    y1 = y0 + tile_size
                    if y1 > h_infer:
                        y1 = h_infer
                    sub_h = y1 - y0
                    if sub_h <= 0:
                        break

                    x0 = 0
                    while x0 < w_infer:
                        x1 = x0 + tile_size
                        if x1 > w_infer:
                            x1 = w_infer
                        sub_w = x1 - x0
                        if sub_w <= 0:
                            break

                        msg.append(f"|| [DEBUG] 處理裁切區塊: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
                        subim0 = im0_infer[y0:y1, x0:x1].copy()

                        # 1) letterbox -> (imgsz, imgsz)
                        sub_img = letterbox(subim0, new_shape=imgsz, auto=False)[0]
                        sub_img = sub_img[:, :, ::-1].transpose(2, 0, 1)
                        sub_img = np.ascontiguousarray(sub_img)

                        sub_img_t = torch.from_numpy(sub_img).to(device)
                        sub_img_t = sub_img_t.half() if half else sub_img_t.float()
                        sub_img_t /= 255.0
                        if sub_img_t.ndimension() == 3:
                            sub_img_t = sub_img_t.unsqueeze(0)

                        # 2) 推論
                        pred_sub = model(sub_img_t, augment=False)[0]
                        pred_sub = rotate_non_max_suppression(
                            pred_sub,
                            conf_thres,
                            iou_thres,
                            classes=[],
                            agnostic=agnostic_nms,
                            theta_format=theta_format,
                            num_extra_outputs=num_extra_outputs,
                            whichdataset=which_dataset
                        )

                        # 3) 若有結果 => 轉回 im0_infer
                        if len(pred_sub) > 0 and pred_sub[0] is not None and len(pred_sub[0]) > 0:
                            det_sub = pred_sub[0]
                            det4pts = xywhtheta24xy_new(det_sub[:, :5])
                            det4pts = r_scale_coords_new((imgsz, imgsz), det4pts, subim0.shape)
                            other_part = det_sub[:, 5:]
                            merged_sub = torch.cat((det4pts, other_part), dim=1)

                            # 加 offset
                            xs = merged_sub[:, [0,2,4,6]] + x0
                            ys = merged_sub[:, [1,3,5,7]] + y0
                            for idx_box in range(4):
                                merged_sub[:, 2*idx_box]   = xs[:, idx_box]
                                merged_sub[:, 2*idx_box+1] = ys[:, idx_box]

                            final_boxes_in_infer.append(merged_sub)

                        # 4) 顯示裁切區域 (debug)
                        if show_crop_visual:
                            # 這裡只顯示 raw subim0，可視需求繪圖
                            cv2.imshow("cropped detection", subim0)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("[DEBUG] 使用者按 'q' -> 中斷裁切 (跳出當前 frame)")
                                # 視需求可改 break/continue/return
                                break

                        # 往右移 step
                        x0 += step
                        if x1 == w_infer:
                            # 已到底 -> break
                            break

                    # 往下移 step
                    y0 += step
                    if y1 == h_infer:
                        # 已到底 -> break
                        break

                # 組合所有區塊結果
                if len(final_boxes_in_infer) > 0:
                    det = torch.cat(final_boxes_in_infer, dim=0)
                else:
                    det = torch.zeros((0, 10), device=device)

                # 若原先有放大，再縮回
                if scale_up_factor != 1.0 and len(det) > 0:
                    det4pts = det[:, :8]
                    other_part = det[:, 8:]
                    det4pts_np = det4pts.cpu().numpy()
                    for i_box in range(det4pts_np.shape[0]):
                        for j in range(0, 8, 2):
                            det4pts_np[i_box, j]   /= scale_up_factor
                            det4pts_np[i_box, j+1] /= scale_up_factor
                    det4pts_t = torch.from_numpy(det4pts_np).to(device)
                    det = torch.cat([det4pts_t, other_part], dim=1)

            # ---------------------------------------------------------
            # 繪製結果到原圖 (im0_original)
            # ---------------------------------------------------------
            if det is not None and len(det):
                # 若為影片則使用 frame 編號
                if dataset.cap:
                    output_8cls_str = str(dataset.frame)
                else:
                    output_8cls_str = str(0)

                for *xyxy, conf, cls in det:
                    # 與原程式一樣檢查是否超出範圍
                    xyxy_t = torch.tensor(xyxy)
                    pts_int = torch.round(xyxy_t).int()

                    if save_vehicle8cls_IOTMOTC_format_txt:
                        if (pts_int < 0).any() or \
                           (pts_int[[0,2,4,6]] >= im0_original.shape[1]).any() or \
                           (pts_int[[1,3,5,7]] >= im0_original.shape[0]).any():
                            continue

                        pts_int = pts_int.tolist()
                        output_8cls_str += ' ' + merged_cls_dict[names[int(cls)]]
                        output_8cls_str += ' {:.2f}'.format(conf)
                        output_8cls_str += ' ' + ' '.join(map(str, pts_int))

                    # 在影像上繪製
                    xyxy_draw = [float(v) for v in xyxy]
                    draw_one_polygon(im0_original, xyxy_draw, int(cls), line_thickness=2)

                # 輸出檔案
                if save_vehicle8cls_IOTMOTC_format_txt:
                    with open(output_detect_result_path, 'a', encoding='utf-8') as file:
                        file.write('{}\n'.format(output_8cls_str))

            # ---------------------------------------------------------
            # 顯示最終結果
            # ---------------------------------------------------------
            if view_img:
                im0_resized = cv2.resize(
                    im0_original,
                    (final_display_width, final_display_height),
                    interpolation=cv2.INTER_LINEAR
                )
                cv2.imshow(str(path), im0_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[DEBUG] 使用者按 'q' -> 結束程式")
                    break
            clear_last_n_lines(prev_line)
            print("\n".join(msg))
            prev_line = len(msg)
        # end for (所有 frame/影像)
        print(f"[DEBUG] Done. 共處理 {frame_count} 個frame, 耗時: {time.time() - t0:.3f}s")

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names, theta_format, \
    save_img_before_nms, which_dataset, save_label_car_tool_format_txt, \
    save_vehicle8cls_IOTMOTC_format_txt, save_dota_txt = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, \
        opt.img_size, opt.cfg, opt.names, opt.theta_format, opt.save_img_before_nms, \
        opt.which_dataset, opt.save_label_car_tool_format_txt, \
        opt.save_vehicle8cls_IOTMOTC_format_txt, opt.save_dota_txt

    if opt.which_dataset == 'dota':
        # dota detect clstxt directory name suffix
        weights_dir_name = weights[0][weights[0].find('exp'):].split(os.sep)[0]
        agnostic_nms_str = ''
        if opt.agnostic_nms: agnostic_nms_str = 'agnostic_nms'
        if opt.augment: augment_str = 'tta'
        else: augment_str = ''
        dota_detect_clstxt_suffix = '{}-{}-conf{}iou{}-{}-{}'.format(weights_dir_name.split('_')[0], weights_dir_name.split('_')[1][:14], opt.conf_thres, opt.iou_thres, agnostic_nms_str, augment_str)

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    num_extra_outputs = get_number_of_extra_outputs(theta_format)

    if save_dota_txt:
        dota_detect_cls_txt_dir = os.getcwd() + os.sep + 'dota_detect_clstxt-' + dota_detect_clstxt_suffix
        if os.path.exists(dota_detect_cls_txt_dir):
            shutil.rmtree(dota_detect_cls_txt_dir)  # delete output folder
        os.makedirs(dota_detect_cls_txt_dir)  # make new output folder 

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz, extra_outs=theta_format, num_extra_outputs=num_extra_outputs).cuda()
    try:
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    except:
        load_darknet_weights(model, weights[0])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer, vid_writer_nms = None, None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = opt.save_img_video
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    avg_inference_nms_t = 0.0
    img_count = 0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        #----------------------
        # draw pred rbox before nms
        if save_img_before_nms:
            img_before_nms = im0s.copy()
            for x in pred:
                x = x[x[:, 4] > opt.conf_thres]  # confidence
    
                if x.shape[0] == 0: continue

                if theta_format.find('dhxdhy') != -1:
                    delta_x, delta_y = x[:,-num_extra_outputs]-x[:,0], x[:,-1]-x[:,1]
                    convert_theta = torch.atan2(-delta_y, delta_x)
                elif theta_format.find('sincos') != -1:
                    convert_theta = torch.atan2(x[:,-num_extra_outputs], x[:,-1])

                convert_theta = torch.where(convert_theta<0,convert_theta+2*math.pi,convert_theta)
                rbox_xywhtheta = torch.cat((x[:,:4], convert_theta.view(-1,1)), dim=1).cpu().numpy()
                all_pts = xywhtheta24xy_new(rbox_xywhtheta)
                # all_pts[:,:] /= gain
                
                all_pts = r_scale_coords_new(img.shape[2:], all_pts, im0s.shape)
                for idx, (pts, conf) in enumerate(zip(all_pts, x[:,4])):
                    # draw_one_rbbox(img_check_boxes, pts.cpu().numpy())
                    draw_one_polygon(img_before_nms, pts, 0)
                    cv2.putText(img_before_nms, '{:.2f}'.format(conf), (int(pts[0]), int(pts[1])), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    if theta_format.find('dxdy') != -1:
                        hx, hy = (pts[0]+pts[2])/2, (pts[1]+pts[3])/2
                        cx, cy = (pts[0]+pts[4])/2, (pts[1]+pts[5])/2
                        cv2.circle(img_before_nms, (int(hx), int(hy)), 2, (0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(img_before_nms, (int(cx), int(cy)), 2, (0,255,0), thickness=-1, lineType=cv2.LINE_AA)
        #----------------------

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, num_extra_outputs=num_extra_outputs)
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, \
                                            theta_format=theta_format, num_extra_outputs=num_extra_outputs, whichdataset=which_dataset)
        t2 = time_synchronized()
        
        img_count += 1
        avg_inference_nms_t += (t2 - t1)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # pred len: (num_img_in_batch, 7[x,y,w,h,θ,conf,classid])
        # pred is a list 

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

            # Write frame ID to vehicle8cls_IOTMOTC_format_txt
            if save_vehicle8cls_IOTMOTC_format_txt:
                # assert dataset.cap, 'Only support video input while saving vehicle8cls_IOTMOTC_format_txt!!'
                if dataset.cap:
                    output_8cls_str = str(dataset.frame)
                else:
                    output_8cls_str = Path(p).name

                # with open(save_path[:save_path.rfind('.')] + '_8cls.txt', 'a') as file:
                #     file.write('{}\n'.format(output_8cls_str))

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Print and write results
            if det is None or len(det) == 0:
                # 如果沒有det(沒東西被偵測到))，還是要產生空的txt檔
                if save_txt:  # Write to file
                    with open(txt_path + '.txt', 'a') as f:
                        pass

                if save_label_car_tool_format_txt:
                    with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                        pass

                if save_dota_txt:
                    with open(dota_detect_cls_txt_dir+os.sep+"Task1_"+names[int(cls)]+".txt", "a+") as file:
                        pass

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # det shape (num_of_det, 10[x1,y1,x2,y2,x3,y3,x4,y4,conf,classid])
                det4pts = xywhtheta24xy_new(det[:,:5])
                det4pts = r_scale_coords(img.shape[2:], det4pts, im0.shape)
                det = torch.cat((det4pts,det[:,5:]), dim=1)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for det_id, (*xyxy, conf, cls) in enumerate(det):  # xyxy: x1y1x2y2x3y3x4y4 clockwise
                    if save_txt:  # Write to file
                        xywh = (fourxy2xywh(torch.tensor(xyxy).view(1, 8))/gn).view(-1)  # normalized xywh
                        dx,dy = xyxy[0]-xyxy[6], xyxy[1]-xyxy[7]
                        theta = torch.atan2(-dy,dx)
                        theta = theta+2*math.pi if theta < 0 else theta
                        xywhtheta = torch.cat((xywh,torch.tensor([theta])),dim=0).tolist()
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywhtheta))  # label format

                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                    if save_label_car_tool_format_txt:
                        xyxy = torch.tensor(xyxy)
                        xywh = (fourxy2xywh(xyxy.view(1, 8))/gn).view(-1)
                        out_theta = torch.atan2(xyxy[7]-xyxy[1], xyxy[6]-xyxy[0])
                        # label_it_car的角點順序是左上角點逆時針 x1y1x2y2x3y3x4y4 -> x1y1x4y4x3y3x2y2
                        pts_roll = torch.roll(xyxy.view(-1,2), -1, 0).flip(0).view(-1)
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            # file.write(('%g ' * 5 + '\n') % (obj_cls, *xywh))  # label format
                            file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(int(cls), *xywh, *pts_roll, out_theta))  # GUI label format

                    if save_vehicle8cls_IOTMOTC_format_txt:
                        xyxy = torch.tensor(xyxy)
                        pts_int = torch.round(xyxy).int()
                        if (pts_int<0).any() or (pts_int[[0,2,4,6]]>=im0.shape[1]).any() or (pts_int[[1,3,5,7]]>=im0.shape[0]).any():
                            pass
                        else:
                            pts_int = pts_int.tolist()
                            output_8cls_str += ' ' + merged_cls_dict[names[int(cls)]]
                            output_8cls_str += ' ' + '{:.2f}'.format(conf) + ' ' + str(pts_int[0]) + ' ' + str(pts_int[1]) \
                                                + ' ' + str(pts_int[2]) + ' ' + str(pts_int[3]) + ' ' + str(pts_int[4]) \
                                                + ' ' + str(pts_int[5]) + ' ' + str(pts_int[6]) + ' ' + str(pts_int[7])

                    if save_dota_txt:
                        img_name_wo_ext = Path(p).name.split('.')[0]
                        pts = torch.tensor(xyxy).tolist()
                        out_dota_str = '%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (img_name_wo_ext, conf,
                                                                                             pts[0], pts[1], pts[2], pts[3],
                                                                                             pts[4], pts[5], pts[6], pts[7]
                                                                                             )
                        
                        with open(dota_detect_cls_txt_dir+os.sep+"Task1_"+names[int(cls)]+".txt", "a+") as file:
                            file.write(out_dota_str)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        tensor_xyxy = torch.tensor(xyxy)
                        # 由於上面用*xyxy從det衷取出八個xy值，會使得*xyxy以list的方式儲存
                        draw_one_polygon(im0, tensor_xyxy, int(cls), line_thickness=2)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        
                        # draw detect id   FONT_HERSHEY_SIMPLEX   FONT_HERSHEY_PLAIN
                        cx = (xyxy[0]+xyxy[4])/2
                        cy = (xyxy[1]+xyxy[5])/2
                        # cv2.rectangle(im0, (int(cx-20),int(cy-20)), (int(cx+30),int(cy+20)), (0,0,0), -1, cv2.LINE_AA)
                        cv2.putText(im0, str(det_id), (int(cx)-3, int(cy)+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                
                show_img = cv2.resize(im0, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                # show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)

                cv2.putText(show_img, p, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow("Detection", show_img)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


            # Save results (image with detections)
            if save_img:
                if save_img_before_nms:
                    extension_idx = -save_path[::-1].find('.')
                    before_nms_save_path = save_path[:extension_idx-1] + '_before_nms' + save_path[extension_idx-1:]

                if dataset.mode == 'images':
                    # Modify
                    # cv2.imwrite(save_path, im0)
                    cv2.imencode('.jpg', im0)[1].tofile(save_path)
                    if save_img_before_nms: 
                        cv2.imencode('.jpg', img_before_nms)[1].tofile(before_nms_save_path)
                        # cv2.imwrite(before_nms_save_path, img_before_nms)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))     
                        if save_img_before_nms:
                            vid_writer_nms = cv2.VideoWriter(before_nms_save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

                    if save_img_before_nms:
                        vid_writer_nms.write(img_before_nms)

            if save_vehicle8cls_IOTMOTC_format_txt:
                with open(save_path[:save_path.rfind('.')] + '_8cls.txt', 'a') as file:
                    file.write('{}\n'.format(output_8cls_str)) 

    if vid_writer is not None: vid_writer.release()
    if vid_writer_nms is not None: vid_writer_nms.release()

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    print('Avg inferene+nms time. (%.3fs)' % (avg_inference_nms_t/img_count))

    # if save_vehicle8cls_IOTMOTC_format_txt:
    #     with open(save_path[:save_path.rfind('.')] + '_8cls.txt', 'r') as infile:
            

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')

    parser.add_argument('--theta-format', type=str, default='dhxdhy', help='theta format to convert (default: dhxdhy)')
    parser.add_argument('--save-img-before-nms', action='store_true', help='update all models')
    parser.add_argument('--which-dataset', type=str, default='vehicle8cls', help='for nms use with same iou thres or different iou thres.(not support)')
    parser.add_argument('--save-label-car-tool-format_txt', action='store_true', help='save results in label_it_car format to *.txt. All results in one txt file')
    parser.add_argument('--save-vehicle8cls-IOTMOTC-format-txt', action='store_true', help='save results in IoT MOTC (運研所) format to *.txt. All results in one txt file')
    parser.add_argument('--save-dota-txt', action='store_true', help='save results in DOTA1.0 format to *.txt.')
    parser.add_argument('--save-img-video', action='store_true', help='save result to images / video')
    

    # 這裡 parse_args() 改成 parse_args(args)，
    # 如果 args 為 None 就抓 sys.argv[1:]
    opt = parser.parse_args(args)
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for w in ['']:
                # 注意：不要再改動 opt.weights，改用 w
                detect(opt)
                strip_optimizer(w)
        else:
            detect(opt)

if __name__ == '__main__':
    main(args=None)