import json
import pandas as pd
from pathlib import Path
import warnings

def convert_box_to_csv(json_path, rename_label, current_id):
    """
    转换X-anylabelingj矩形框标注为符合deepforest规范的.csv
    
    参数:
    json_path :  标注json地址图像列表
    rename_label :  重命名标签
    current_id :  标识id
    返回：
    pd.DataFrame对象
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        shapes = data.get('shapes')
    except KeyError:
        warnings.warn(f"{json_path}中未找到'shapes'键.", UserWarning)
        return  None
    all_data = []
    for shape in shapes:
        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        if rename_label is None:
            label = shape['label']
        else:
            label = rename_label
        
        all_data.append({
            "ID": current_id,
            "image_path": data.get("imagePath", ""), 
            "xmin": min(x_coords),
            "ymin": min(y_coords),
            "xmax": max(x_coords),
            "ymax": max(y_coords),
            "label": label,
            "shape_type": shape.get("shape_type", ""),
            "file_name": Path(json_path).name # 添加文件名以便追踪     
        })
    return pd.DataFrame(all_data)

def convert_polygon_to_csv():
    pass

if __name__ == "__main__":

    from deepforest.visualize import plot_results
    from deepforest import utilities

    test_df = convert_box_to_csv("./test_img/crop_601.json", 
    rename_label = "Tree", current_id = 1)
    test_df.to_csv("./test_img/crop_601.csv", index = False)
    converted_df = utilities.read_file("./test_img/crop_601.csv")
    print(converted_df)
    # 当image_path为文件名时，plot_results会在当前目录下寻找图片文件，若无法找到会报错
    plot_results(converted_df)