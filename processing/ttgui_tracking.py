import cv2
import math
import numpy as np
import csv
from collections import Counter
from Model.sort8 import SORT
from shapely.geometry import Polygon
from tqdm import tqdm

def order_quad_ccw(pts: np.ndarray) -> np.ndarray:
    """將 4 點按逆時針排序，避免自交。pts shape=(4,2)"""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    return pts[np.argsort(ang)]

def overlap_ratio_by_cv(quad_i: np.ndarray, quad_j: np.ndarray) -> float:
    """
    回傳兩個凸四邊形之交集面積 / quad_i 面積。
    quad_* shape=(4,2)，數值型別可為 float32/float64。
    內含 AABB 快篩，無相交直接回 0。
    """
    # AABB 快篩（幾乎零成本）
    xi0, yi0 = np.min(quad_i, axis=0); xi1, yi1 = np.max(quad_i, axis=0)
    xj0, yj0 = np.min(quad_j, axis=0); xj1, yj1 = np.max(quad_j, axis=0)
    if xi1 < xj0 or xj1 < xi0 or yi1 < yj0 or yj1 < yi0:
        return 0.0

    area_i = float(cv2.contourArea(quad_i.astype(np.float32)))
    if area_i <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(
        quad_i.astype(np.float32), quad_j.astype(np.float32)
    )
    return float(inter_area) / area_i if area_i > 0 else 0.0

def RT_tracking_name():
    import os
    # 只取出檔名（包括副檔名）
    file_name = os.path.basename(__file__)
    # 移除副檔名並回傳
    return os.path.splitext(file_name)[0]
def RT_sort_name():
    return SORT.RT_sort_name()

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines

def parse_line(line):
    data = list(map(float, line.split()))
    frame_no = int(data[0])
    rects = []
    for i in range(1, len(data), 10):
        class_id = int(data[i])
        conf = data[i+1]
        points = np.array(data[i+2:i+10]).reshape(4, 2)
        rects.append((class_id, conf, points))
    return frame_no, rects

def dist(p1, p2):
    return math.sqrt(sum((p2[i] - p1[i])**2 for i in range(len(p1))))

def rotated_rect_to_5_params(points2):
    points = np.array(points2, dtype=np.int32).reshape(4, 2)
    rect2 = cv2.minAreaRect(points)
    (cx, cy), (w, h), a = rect2
    # h = math.dist(points[0], points[1]) those code only work in python 3.8 or higher version cause math.dist.
    # w = math.dist(points[0], points[3]) current version is 3.6.7
    h = dist(points[0], points[1])
    w = dist(points[0], points[3])
    
    if points[0][0] == min(points[0][0],points[1][0],points[2][0],points[3][0]):
        a = 180 - a
        if points[0][0] == points[1][0]:
            a = 180
    elif points[0][1] == min(points[0][1],points[1][1],points[2][1],points[3][1]):
        a = 90 - a
        if points[0][1] == points[1][1]:
            a = 90          
    elif points[0][1] == max(points[0][1],points[1][1],points[2][1],points[3][1]):
        a = 270 - a
        if points[0][1] == points[1][1]:
            a = 270      
    elif points[0][0] == max(points[0][0],points[1][0],points[2][0],points[3][0]):
        a = 360 - a
        if points[0][0] == points[1][0]:
            a = 0        
    return np.array([cx, cy, w, h, a], dtype=np.float32)

def add_trajectory(dict_IDs, ID, frame, style, pos):
    if ID not in dict_IDs:
        # 初始化一個新的軌跡列表，列表的第一個元素是 ID，第二個元素是 frame，之後是位置
        dict_IDs[ID] = [ID, frame, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]
    
    # 添加新的位置到軌跡列表並更新 frame
    dict_IDs[ID][11].extend([pos])
    dict_IDs[ID][2] = frame
    dict_IDs[ID][3+style] = dict_IDs[ID][3+style] + 1
    return dict_IDs

def interpolate_zeros(pos_list):
    # 找到所有連續零值的範圍（開始和結束索引）
    zero_ranges = []
    start_index = None
    for i in range(1, len(pos_list)-1):
        if np.all(pos_list[i] == 0):  # 使用 numpy 的 all() 來檢查陣列中的所有元素
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                zero_ranges.append((start_index, i))
                start_index = None

    # 如果我們在列表的末尾找到零值，我們需要手動添加這個範圍
    if start_index is not None:
        zero_ranges.append((start_index, len(pos_list)))

    # 檢查最後一個零值範圍是否延伸到列表的末尾，如果是的話，刪除這些元素
    if zero_ranges and zero_ranges[-1][1] == len(pos_list):
        pos_list = pos_list[:zero_ranges[-1][0]]
        zero_ranges = zero_ranges[:-1]

    # 對於每個零值範圍，計算開始索引和結束索引之間的位置的插值
    zero_ratio = 0
    for start_index, end_index in zero_ranges:
        start_pos = np.array(pos_list[start_index - 1])
        end_pos = np.array(pos_list[end_index])

        # 計算插值的增量
        increment = (end_pos - start_pos) / (end_index - start_index + 1)
        zero_ratio = zero_ratio + end_index - start_index + 1

        # 替換零值
        for i in range(start_index, end_index):
            pos_list[i] = np.round(start_pos + (i - start_index + 1) * increment).astype(int) # 使用 numpy 的 round 函數四捨五入
    
    if len(pos_list) > 20:
        zero_ratio = zero_ratio/len(pos_list)
    else:
        zero_ratio = 1
        
    return pos_list, zero_ratio

# 定義一個函數來計算兩個點之間的歐氏距離
def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def to_points(r):
    # 將一維的座標列表轉換為二維點的列表
    return [(r[i], r[i+1]) for i in range(0, len(r), 2)]

def min_distance(r1, r2):
    # 將 r1 和 r2 轉換為二維點的列表
    r1_points = to_points(r1)
    r2_points = to_points(r2)

    # 計算所有可能的旋轉
    # rotations = [r2_points[i:] + r2_points[:i] for i in range(len(r2_points))]
    rotations = [r2_points[i:] + r2_points[:i] for i in range(0, len(r2_points), 2)]

    # 計算每種旋轉對應的總位移
    distances = [sum(distance(p1, p2) for p1, p2 in zip(r1_points, rotated_r2)) for rotated_r2 in rotations]
    
    # 找出最小總位移及其對應的旋轉
    min_dist, min_rotation = min(zip(distances, rotations))

    # 轉回一維列表
    min_rotation_idx = rotations.index(min_rotation)
    # min_rotation = [coord for point in min_rotation for coord in point]
    
    min_rotation = list(np.array(min_rotation).flatten())
    
    return min_rotation, int(min_rotation_idx)

def trace_reorder(data):
    # 初始化新的數據和第一點位置的記錄器
    new_data = [data[0].tolist()]  # 第一個矩形的數據不變
    first_point_positions = [0]
    zero_indices = []

    i = 1
    # 對於每一對相鄰的矩形，找出使總位移最小的旋轉方式
    for i in range(1, len(data)):
        if np.all(data[i]==0):
            zero_indices.append(i)
            continue
        r1 = new_data[-1]
        r2 = list(data[i])
        min_rotation, position = min_distance(r1, r2)
           
        new_data.append(min_rotation)
        first_point_positions.append(position)
           
    # 找出出現次數最多的第一點位置
    counter = Counter(first_point_positions)
    most_common_position = counter.most_common(1)[0][0]

    # 根據出現次數最多的第一點位置調整所有矩形的角點順序
    final_data = [r[most_common_position*4:] + r[:most_common_position*4] for r in new_data]

    # 車頭與移動方向不一致
    i = 0
    cum_move = 0   
    for i in range(0, len(final_data)-1):
        cum_move = cum_move + (final_data[i+1][0]-final_data[i][0])*np.sign(final_data[i][0]-final_data[i][4])
        cum_move = cum_move + (final_data[i+1][1]-final_data[i][1])*np.sign(final_data[i][1]-final_data[i][5])     
    if cum_move < 0:
        final_data = [r[4:] + r[:4] for r in final_data]

    # 將所有點為0的矩形添加回去
    for i in zero_indices:
        final_data.insert(i, [0]*8)

    return final_data

def del_outliers(quadrilateral_positions, limited_angle):
    # 計算四邊形的中心點
    def calculate_center(points):
        x_coords = points[0::2]
        y_coords = points[1::2]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        return center_x, center_y

    # 計算角度
    def calculate_angle(center, point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        rad = np.arctan2(dy, dx)
        deg = np.degrees(rad)
        if deg < 0:
            deg += 360
        return deg

    # 計算每個四邊形的角度
    angles = []
    for quad in quadrilateral_positions:
        center = calculate_center(quad)
        point = (quad[0], quad[1])  # (x1, y1)
        angle = calculate_angle(center, point)
        angles.append(angle)

    # 根據角度數據更新四邊形的座標
    n = len(angles)
    new_positions = quadrilateral_positions.copy()
    i = 0
    for i in range(n):
        if i == 0 or np.all(quadrilateral_positions[i] == 0):
            continue
        diff_angle_prev = abs(angles[i] - angles[i-1])
        diff_angle_prev = min(360 - diff_angle_prev, diff_angle_prev)
        # if diff_angle_prev > limited_angle or np.all(quadrilateral_positions[i-1] == 0):  # 使用 numpy 的 all() 來檢查陣列中的所有元素
        if diff_angle_prev > limited_angle:  # 使用 numpy 的 all() 來檢查陣列中的所有元素
            new_positions[i] = [0, 0, 0, 0, 0, 0, 0, 0]
            continue
        if i < n-1:
            diff_angle_next = abs(angles[i] - angles[i+1])
            diff_angle_next = min(360 - diff_angle_next, diff_angle_next)
            # if diff_angle_next > limited_angle or np.all(quadrilateral_positions[i+1] == 0):
            if diff_angle_next > limited_angle:
                new_positions[i] = [0, 0, 0, 0, 0, 0, 0, 0]

    return new_positions

def process_trajectory(quadrilateral_positions):
    # 計算四邊形的中心點並生成其軌跡
    def calculate_center(quadrilateral):
        x = np.mean([quadrilateral[i] for i in range(0, len(quadrilateral), 2)])
        y = np.mean([quadrilateral[i] for i in range(1, len(quadrilateral), 2)])
        return np.array([x, y])

    centers = np.array([calculate_center(quad) for quad in quadrilateral_positions])

    # 計算每個四邊形的最小邊長
    def calculate_min_side_length(quadrilateral):
        points = np.array(quadrilateral).reshape(-1, 2)
        side_lengths = [np.linalg.norm(points[i] - points[(i+1)%4]) for i in range(4)]
        return min(side_lengths)

    min_side_lengths = [calculate_min_side_length(quad) for quad in quadrilateral_positions]

    # 根據方向生成新的正方形
    def generate_square(center, side_length, angle):
        dx = side_length / 2 * np.array([np.cos(angle), np.sin(angle)])
        dy = np.array([-dx[1], dx[0]])
        return np.array([center + dx - dy, center + dx + dy, center - dx + dy, center - dx - dy]).flatten()

    new_positions = []
    for i, center in enumerate(centers):
        # 往前找到距離最遠但不超過10的中心點位置p_before
        j = i
        while j > 0 and np.linalg.norm(center - centers[j]) < 10:
            j -= 1

        # 往後找到距離最遠但不超過10的中心點位置p_after
        k = i
        while k < len(centers)-1 and np.linalg.norm(center - centers[k]) < 10:
            k += 1

        # 計算p_before連往p_after的方向
        if j >= 0 and k < len(centers):
            direction = centers[k] - centers[j]
            angle = np.arctan2(direction[1], direction[0])
        else:
            angle = 0  # 邊界條件，若找不到p_before或p_after則方向為0

        # 生成新的正方形並將四個角點整數化
        new_positions.append(np.round(generate_square(center, min_side_lengths[i], angle)).astype(int))

    return new_positions

def main(yolo_txt, tracking_csv, show, display_callback, trk1_set=(10, 2, 0.05), trk2_set=(10, 2, 0.1) , stab_video=None):
    lines = read_file(yolo_txt)

    if show:
        cap = cv2.VideoCapture(stab_video)

    trackers_1 = SORT(trk1_set[0], trk1_set[1], trk1_set[2]) # 小車 (人、機車、自行車)
    trackers_2 = SORT(trk2_set[0], trk2_set[1], trk2_set[2]) # 大車 (汽車、卡車、公車)

    track1 = {} 
    track2 = {}

    # Wrap the loop with tqdm to display progress
    for line in tqdm(lines, desc="Processing frames", total=len(lines)):
        frame_no, rects_raw = parse_line(line)

        if show:
            ret, frame = cap.read()

        polygons = []
        for rect in rects_raw:
            class_id, conf, points = rect            
            points = points.astype(np.int32)
            polygons.append(Polygon(points.reshape(4,2)))

        # 以 OpenCV 計交集面積，移除 shapely
        quads = []
        for class_id, conf, points in rects_raw:
            pts = points.astype(np.float32).reshape(4, 2)
            pts = order_quad_ccw(pts)                 # 順序校正，避免自交
            quads.append(pts)

        # Use loop to check and remove rectangles with large intersection
        rects = []
        for i, rect in enumerate(rects_raw):
            # 面積為 0 直接跳過
            area_i = cv2.contourArea(quads[i])
            if area_i <= 0:
                continue

            if i == 0:
                rects.append(rect)
                continue

            # 只要與任一先前四邊形的交集 / 自身面積 > 0.5 就丟掉
            cond = not any(
                overlap_ratio_by_cv(quads[i], quads[j]) > 0.5
                for j in range(i)
                # AABB 已在 overlap_ratio_by_cv 內做過，這裡不需再判
            )
            if cond:
                rects.append(rect)

        group1 = []
        group2 = []
        for rect in rects:
            class_id, conf, points = rect
            points = points.astype(np.int32)

            rect_5_params = rotated_rect_to_5_params(points)
            fullparams = np.concatenate((np.array(rect_5_params).flatten('C'), np.array(points).flatten('C'), [class_id]), axis=0)

            if class_id in [0, 1, 2]:
                group1.append(fullparams)
            else:
                group2.append(fullparams)

        tracked_objects_1 = trackers_1.update(np.array(group1))
        tracked_objects_2 = trackers_2.update(np.array(group2))

        for track_id, rect, rect2, vote in tracked_objects_1:
            box2 = cv2.boxPoints([(rect[0], rect[1]), (rect[2], rect[3]), 180-rect[4]])
            box = np.intp(box2) 

            rect3 = rect2.astype(int).reshape(-1, 2)
            if show:
                cv2.line(frame, rect3[0], rect3[1], (0, 0, 255), 2)
                cv2.polylines(frame, [rect3], True, (128, 128, 255), 3)
                cv2.putText(frame, str(track_id), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            track1 = add_trajectory(track1, track_id, frame_no, vote, rect3.reshape(1, 8))

        for track_id, rect, rect2, vote in tracked_objects_2:
            box2 = cv2.boxPoints([(rect[0], rect[1]), (rect[2], rect[3]), 180-rect[4]])
            box = np.intp(box2) 

            rect3 = rect2.astype(int).reshape(-1, 2)
            if show:
                cv2.line(frame, rect3[0], rect3[1], (0, 0, 255), 2)
                cv2.polylines(frame, [rect3], True, (128, 128, 255), 3)
                cv2.putText(frame, str(track_id), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            track2 = add_trajectory(track2, track_id, frame_no, vote, rect3.reshape(1, 8))

        if show:
            frame2_resized = cv2.resize(frame, (960, 540), cv2.INTER_AREA)
            # cv2.imshow('Frame', frame2_resized)
            display_callback(frame2_resized)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Output CSV file
    V_type = "pumctbhg" 
    with open(tracking_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        i = 0
        for id, data in track2.items():
            print(f"track2 : [{id}]/[{len(track2)}]", end='\r')
            if len(data[11]) > 20:
                traj = np.array(data[11]).reshape(-1, 8)               
                traj = np.array(del_outliers(traj, 5)).reshape(-1, 8)
                reorder_data = np.array(trace_reorder(traj)).reshape(-1, 8)              
                traj, zero_ratio = interpolate_zeros(reorder_data)
                if zero_ratio < 0.4:
                    writer.writerow([i, data[1], data[1]+len(traj)-1, 'X', 'X', V_type[np.argmax(data[3:11])]] + list(np.array(traj).flatten()) )
                    i += 1
                
        for id, data in track1.items():
            print(f"track1 : [{id}]/[{len(track1)}]", end='\r')
            if len(data[11]) > 20:
                if V_type[np.argmax(data[3:11])] == 'p':
                    traj = np.array(data[11]).reshape(-1, 8)    
                    traj, zero_ratio = interpolate_zeros(traj)                    
                    traj = process_trajectory(traj)
                    if zero_ratio < 0.8:
                        writer.writerow([i, data[1], data[1]+len(traj)-1, 'X', 'X', V_type[np.argmax(data[3:11])]] + list(np.array(traj).flatten()) )
                        i += 1
                    continue
                traj = np.array(data[11]).reshape(-1, 8)               
                traj = np.array(del_outliers(traj, 15)).reshape(-1, 8)
                reorder_data = np.array(trace_reorder(traj)).reshape(-1, 8)
                traj, zero_ratio = interpolate_zeros(reorder_data)
                if zero_ratio < 0.7:
                    writer.writerow([i, data[1], data[1]+len(traj)-1, 'X', 'X', V_type[np.argmax(data[3:11])]] + list(np.array(traj).flatten()) )
                    i += 1
    if show:
        cap.release()    
    csvfile.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

