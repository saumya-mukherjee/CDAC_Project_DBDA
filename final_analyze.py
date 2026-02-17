import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression

# This file contains all functions. No 'use.py' is needed.

# --- 1. Non-Max Suppression (Helper Function from your 'use.py') ---
def non_max_suppression(boxes, overlap_thresh=0.2):
    """
    Applies non-maximum suppression to a list of bounding boxes.
    (This function was in your 'use.py' file)
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")

# --- 2. Sentence Alignment (Your Original Function) ---
def sentence_alignment(img):
    """
    Analyzes the alignment (upward, downward, straight) of text lines.
    (This is from your 'analyze.py')
    """
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))
    dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10 and w * h < 40000:
            boxes.append((x, y, w, h))

    # Apply NMS
    boxes = non_max_suppression(boxes, overlap_thresh=0.2)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    if not boxes:
        return original, [], 0.05

    # Cluster boxes into lines
    y_centers = np.array([[y + h//2] for (x, y, w, h) in boxes])
    db = DBSCAN(eps=30, min_samples=1).fit(y_centers)
    labels = db.labels_

    lines = {}
    for label, box in zip(labels, boxes):
        lines.setdefault(label, []).append(box)

    results = []
    line_number = 1
    all_slopes = []
    for label in lines:
        line_boxes = sorted(lines[label], key=lambda b: b[0])
        if len(line_boxes) < 2:
            continue
        x_coords = np.array([x + w//2 for (x, y, w, h) in line_boxes]).reshape(-1, 1)
        y_coords = np.array([y + h//2 for (x, y, w, h) in line_boxes])
        reg = LinearRegression().fit(x_coords, y_coords)
        all_slopes.append(reg.coef_[0])

    if all_slopes:
        slope_std = np.std(all_slopes)
        straight_threshold = max(0.02, slope_std * 0.5)
    else:
        straight_threshold = 0.05

    for label in sorted(lines.keys()):
        line_boxes = sorted(lines[label], key=lambda b: b[0])
        if len(line_boxes) < 2:
            continue
        x_coords = np.array([x + w//2 for (x, y, w, h) in line_boxes]).reshape(-1, 1)
        y_coords = np.array([y + h//2 for (x, y, w, h) in line_boxes])
        reg = LinearRegression().fit(x_coords, y_coords)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        if abs(slope) < straight_threshold:
            direction = "straight"
        elif slope < 0: # In image coordinates, negative slope is upward
            direction = "upward"
        else:
            direction = "downward"

        results.append({
            'line_number': line_number,
            'slope': slope,
            'direction': direction,
            'word_count': len(line_boxes)
        })

        # Draw regression line
        x_min = min(x_coords)[0]
        x_max = max(x_coords)[0]
        y_min_pred = int(slope * x_min + intercept)
        y_max_pred = int(slope * x_max + intercept)

        color = (255, 0, 0) if direction == "straight" else (0, 0, 255) if direction == "upward" else (0, 255, 0)
        cv2.line(original, (x_min, y_min_pred), (x_max, y_max_pred), color, 2)
        line_number += 1

    return original, results, straight_threshold

# --- 3. Tall & Narrow Letters (Your Original Function) ---
def detect_tall_narrow_letters(img):
    """
    (This is from your 'analyze.py')
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h = img.shape[0]
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < img_h * 0.05:
            continue
        aspect_ratio = w / h
        if aspect_ratio < (1/3.0):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
    return img, count

# --- 4. Writing Pressure (Your Original Function) ---
def detect_writing_pressure(img):
    """
    (This is from your 'analyze.py')
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    stroke_thickness = np.mean(cv2.reduce(binary, 0, cv2.REDUCE_AVG)) / 255
    stroke_thickness = round(float(stroke_thickness), 2)

    text_pixels = gray[gray < 200]  # exclude light background
    intensity = np.mean(text_pixels) if text_pixels.size > 0 else 255
    intensity = round(float(intensity), 2)

    pressure = "Heavy" if (stroke_thickness > 2.5 or intensity < 100) else "Light"

    return pressure, stroke_thickness, intensity

# --- 5. Text Alignment (Your Original Function) ---
def detect_alignment(img_gray):
    """
    Analyzes margin alignment.
    (This is from your 'analyze.py', receives grayscale)
    """
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return "No text detected"

    avg_x = np.mean(xs)
    img_center = binary.shape[1] / 2

    if avg_x < img_center * 0.9:  # Closer to left
        return "Left Aligned"
    elif avg_x > img_center * 1.1: # Closer to right
        return "Right Aligned"
    else:                          # Near center
        return "Center Aligned"
    
# --- 6. Handwriting Size (Your Original Function) ---
def detect_handwriting_size(img_gray):
    """
    Analyzes the overall size of letters.
    (This is from your 'analyze.py', receives grayscale)
    """
    gray = img_gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    heights = []
    for stat in stats[1:]:  # skip background
        x, y, w, h, area = stat
        if h > 8 and w > 5:  # filter noise
            heights.append(h)

    if not heights:
        return "No handwriting detected" # Only returns one value

    median_height = np.median(heights)
    ratio = median_height / binary.shape[0]

    # This is your original logic, which doesn't include "Medium"
    if ratio < 0.035:
        return "Small Handwriting"
    else:
        return "Large Handwriting"

# --- 7. Letter Slant (NEW Enhanced Function) ---
def detect_letter_slant(img):
    """
    Analyzes the vertical slant (left, right, vertical) of letters.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 15 and w > 3 and h/w > 1.2: # Taller than wide
            try:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rect
                
                if width < height:
                    angle = angle - 90
                
                if abs(angle) < 45 and abs(angle) > 2:
                    angles.append(angle)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            except:
                continue

    if not angles:
        return img, "Slant not detected", 0

    avg_angle = np.mean(angles)
    
    if avg_angle > 5:
        slant_category = "Right Slant"
    elif avg_angle < -5:
        slant_category = "Left Slant"
    else:
        slant_category = "Vertical"
        
    return img, slant_category, avg_angle

# --- 8. Word Spacing (NEW Enhanced Function) ---
def detect_word_spacing(img):
    """
    Analyzes the average horizontal distance between words.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            boxes.append((x, y, w, h))
            
    if len(boxes) < 2:
        return img, "Insufficient words", 0

    boxes.sort(key=lambda b: (b[1], b[0]))

    distances = []
    avg_word_width = np.mean([w for (x, y, w, h) in boxes])
    
    y_centers = np.array([[y + h//2] for (x, y, w, h) in boxes])
    db = DBSCAN(eps=20, min_samples=1).fit(y_centers)
    labels = db.labels_

    lines = {}
    for label, box in zip(labels, boxes):
        lines.setdefault(label, []).append(box)

    for label in sorted(lines.keys()):
        line_boxes = sorted(lines[label], key=lambda b: b[0])
        for i in range(len(line_boxes) - 1):
            curr_box_end = line_boxes[i][0] + line_boxes[i][2]
            next_box_start = line_boxes[i+1][0]
            
            distance = next_box_start - curr_box_end
            if distance > 0: 
                distances.append(distance)
            
            x1 = curr_box_end
            y1 = line_boxes[i][1] + line_boxes[i][3] // 2
            x2 = next_box_start
            y2 = y1
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not distances:
        return img, "Spacing not detected", 0

    avg_distance = np.mean(distances)
    
    if avg_distance > avg_word_width * 0.8:
        spacing_category = "Wide"
    elif avg_distance < avg_word_width * 0.3:
        spacing_category = "Narrow"
    else:
        spacing_category = "Consistent"
        
    return img, spacing_category, avg_distance

# --- 9. Line Spacing (NEW Enhanced Function - FIXED) ---
def detect_line_spacing(img):
    """
    Analyzes the vertical distance between text lines.
    (This is the function that was broken)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- FIX: Changed MORPH_Rect to cv2.MORPH_RECT ---
    # This was the cause of the NameError in your screenshot.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            boxes.append((x, y, w, h))
            
    if len(boxes) < 2:
        return img, "Insufficient lines", 0
        
    y_centers = np.array([[y + h//2] for (x, y, w, h) in boxes])
    db = DBSCAN(eps=20, min_samples=1).fit(y_centers)
    labels = db.labels_

    line_boxes = {}
    for label, box in zip(labels, boxes):
        x, y, w, h = box
        if label not in line_boxes:
            line_boxes[label] = [x, y, x + w, y + h] # min_x, min_y, max_x, max_y
        else:
            line_boxes[label][0] = min(line_boxes[label][0], x)
            line_boxes[label][1] = min(line_boxes[label][1], y)
            line_boxes[label][2] = max(line_boxes[label][2], x + w)
            line_boxes[label][3] = max(line_boxes[label][3], y + h)

    sorted_lines = sorted(line_boxes.values(), key=lambda b: b[1])
    
    distances = []
    line_heights = []
    for i in range(len(sorted_lines) - 1):
        curr_line_bottom = sorted_lines[i][3]
        next_line_top = sorted_lines[i+1][1]
        
        distance = next_line_top - curr_line_bottom
        if distance > 0: 
            distances.append(distance)
        
        line_height = sorted_lines[i][3] - sorted_lines[i][1]
        line_heights.append(line_height)
        
        x1 = sorted_lines[i][0]
        y1 = curr_line_bottom
        x2 = sorted_lines[i][0]
        y2 = next_line_top
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not distances or not line_heights:
        return img, "Spacing not detected", 0

    avg_distance = np.mean(distances)
    avg_line_height = np.mean(line_heights)

    if avg_distance > avg_line_height * 1.5:
        spacing_category = "Wide"
    elif avg_distance < avg_line_height * 0.5:
        spacing_category = "Narrow"
    else:
        spacing_category = "Consistent"
        
    return img, spacing_category, avg_distance

# --- 10. T-Bar Analysis (NEW Enhanced Function) ---
def detect_t_bars(img):
    """
    Finds t-stems and analyzes the placement of the t-bar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    stems = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    bars = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    stem_contours, _ = cv2.findContours(stems, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bar_contours, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    placements = []
    t_bars_found = 0
    
    for stem_cnt in stem_contours:
        sx, sy, sw, sh = cv2.boundingRect(stem_cnt)
        if sh < 15 or sh > 100: continue 
        
        stem_center_x = sx + sw // 2
        
        for bar_cnt in bar_contours:
            bx, by, bw, bh = cv2.boundingRect(bar_cnt)
            if bw < 5 or bw > 50: continue 
            
            bar_center_y = by + bh // 2
            
            if (sx < bx + bw and sx + sw > bx) and \
               (sy - 5 < bar_center_y < sy + sh * 0.8): 
                
                t_bars_found += 1
                
                placement_ratio = (bar_center_y - sy) / sh
                
                if placement_ratio < 0.33:
                    placements.append("High on Stem")
                elif placement_ratio < 0.66:
                    placements.append("Middle of Stem")
                else:
                    placements.append("Low on Stem")
                    
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2) # Stem
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2) # Bar
                break 
    
    if t_bars_found == 0 or not placements:
        dominant_placement = "N/A"
    else:
        dominant_placement = max(set(placements), key=placements.count)
        if placements.count(dominant_placement) < len(placements) * 0.5:
             dominant_placement = "Variable" 
             
    return img, {"t_bars_found": t_bars_found, "dominant_placement": dominant_placement}

# --- 11. I-Dot Analysis (NEW Enhanced Function) ---
def detect_i_dots(img):
    """
    Finds 'i' stems and analyzes the placement and shape of their dots.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stems = []
    dots = []
    
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 5]
    if not heights:
         return img, {"i_dots_found": 0, "dominant_placement": "N/A", "dominant_shape": "N/A"}
         
    heights_arr = np.array(heights).reshape(-1, 1)
    if len(heights) < 2:
        middle_zone_height = np.median(heights)
    else:
        try:
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(heights_arr)
            middle_zone_height = min(kmeans.cluster_centers_.flatten())
        except ValueError:
            middle_zone_height = np.median(heights)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if (middle_zone_height * 0.1 < h < middle_zone_height * 0.5) and \
           (middle_zone_height * 0.1 < w < middle_zone_height * 0.5) and area < 100:
            dots.append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w//2, 'cy': y + h//2, 'cnt': cnt})
            
        elif (middle_zone_height * 0.7 < h < middle_zone_height * 1.3) and \
             (w < h * 0.6) and w < middle_zone_height * 0.5:
            stems.append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w//2, 'cy': y + h//2})

    placements = []
    shapes = []
    i_dots_found = 0
    
    for stem in stems:
        best_dot = None
        min_dist = middle_zone_height 
        
        for dot in dots:
            if dot['cy'] < stem['y']:
                dx = abs(dot['cx'] - stem['cx'])
                dy = stem['y'] - dot['cy']
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dx < stem['w'] * 3 and dy < min_dist:
                    min_dist = dy
                    best_dot = dot
        
        if best_dot:
            i_dots_found += 1
            cv2.rectangle(img, (stem['x'], stem['y']), (stem['x'] + stem['w'], stem['y'] + stem['h']), (255, 0, 0), 2) # Stem
            cv2.rectangle(img, (best_dot['x'], best_dot['y']), (best_dot['x'] + best_dot['w'], best_dot['y'] + best_dot['h']), (0, 255, 0), 2) # Dot
            
            dx = best_dot['cx'] - stem['cx']
            dy = stem['y'] - best_dot['cy']
            
            if abs(dx) < stem['w']: 
                if dy > middle_zone_height * 0.8:
                    placements.append("Far Above")
                else:
                    placements.append("Precisely Above")
            elif dx > 0:
                placements.append("To the Right")
            else:
                placements.append("To the Left")
            
            area = best_dot['w'] * best_dot['h']
            aspect_ratio = best_dot['w'] / best_dot['h']
            
            if area < 15 and aspect_ratio > 2.0:
                shapes.append("Slash")
            elif aspect_ratio > 0.7 and aspect_ratio < 1.3:
                shapes.append("Circle")
            else:
                shapes.append("Dot")

    if i_dots_found == 0:
        return img, {"i_dots_found": 0, "dominant_placement": "N/A", "dominant_shape": "N/A"}

    dominant_placement = max(set(placements), key=placements.count) if placements else "N/A"
    dominant_shape = max(set(shapes), key=shapes.count) if shapes else "N/A"
    
    return img, {"i_dots_found": i_dots_found, "dominant_placement": dominant_placement, "dominant_shape": dominant_shape}