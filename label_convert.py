import os
import json
import pandas as pd
from typing import Union, List

def convert_box_to_pd(json_data, rename_label=None, current_id=0):
    """
    专门处理矩形标注的转换函数
    
    参数:
    json_data : dict - 已解析的JSON对象（必须全部是矩形标注）
    rename_label : str/None - 重命名标签
    current_id : int - 起始ID
    返回:
    pd.DataFrame - 包含列: ID, image_path, xmin, ymin, xmax, ymax, label, shape_type
    """
    records = []
    
    for shape in json_data["shapes"]:
        if shape["shape_type"] != "rectangle":
            raise ValueError("JSON文件中包含非矩形标注，请使用convert_polygon_to_pd处理")
            
        # 提取矩形坐标（从4个点中计算边界框）
        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        records.append({
            "ID": current_id,
            "image_path": json_data["imagePath"],
            "xmin": min(x_coords),
            "ymin": min(y_coords),
            "xmax": max(x_coords),
            "ymax": max(y_coords),
            "label": rename_label if rename_label is not None else shape["label"],
            "shape_type": "rectangle"
        })
        current_id += 1
    
    return pd.DataFrame(records)

def convert_polygon_to_pd(json_data, rename_label=None, current_id=0):
    """
    专门处理多边形标注的转换函数
    
    参数:
    json_data : dict - 已解析的JSON对象（必须全部是多边形标注）
    rename_label : str/None - 重命名标签
    current_id : int - 起始ID
    
    返回:
    pd.DataFrame - 包含列: ID, image_path, polygons, label, shape_type
    """
    records = []
    
    for shape in json_data["shapes"]:
        if shape["shape_type"] != "polygon":
            raise ValueError("JSON文件中包含非多边形标注，请使用convert_box_to_pd处理")
            
        records.append({
            "ID": current_id,
            "image_path": json_data["imagePath"],
            "polygons": shape["points"],
            "label": rename_label if rename_label is not None else shape["label"],
            "shape_type": "polygon"
        })
        current_id += 1
    
    return pd.DataFrame(records)


def convert_anylabeling_to_pd(json_data, rename_label=None, current_id=0):
    """
    自动判断标注类型并调用对应的转换函数
    参数:
    json_data : dict - 已解析的JSON对象
    rename_label : str/None - 重命名标签
    current_id : int - 起始ID
    
    返回:
    pd.DataFrame - 根据标注类型返回对应的DataFrame
    
    异常:
    ValueError - 当JSON中没有shapes或包含混合类型时抛出
    """
    if not json_data.get("shapes"):
        # 无shapes返回空表
        return pd.DataFrame()
    
    # 获取所有shape的类型集合
    shape_types = {shape["shape_type"] for shape in json_data["shapes"]}
    
    # 检查是否为单一类型
    if len(shape_types) > 1:
        raise ValueError("JSON文件中包含混合标注类型，请分开处理")
    if not shape_types:
        return pd.DataFrame()
    
    # 根据类型调用对应函数
    shape_type = shape_types.pop()
    if shape_type == "rectangle":
        return convert_box_to_pd(json_data, rename_label, current_id)
    elif shape_type == "polygon":
        return convert_polygon_to_pd(json_data, rename_label, current_id)
    else:
        raise ValueError(f"不支持的标注类型: {shape_type}")
    

def process_json_files(
    json_dir: str,
    start_id: int = 0,
    combine: bool = True,
    output_dir: str = None,
    write_csv: bool = False,
    rename_label: str = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    批量处理目录下的JSON标注文件
    
    参数:
    json_dir : str - JSON文件目录路径
    start_id : int - 起始ID编号
    combine : bool - 是否合并为一个DataFrame
    output_dir : str - CSV输出目录（write_csv=True时必须提供）
    write_csv : bool - 是否输出CSV文件
    rename_label : str - 统一重命名标签（可选）
    
    返回:
    根据combine参数返回：
    - True: 单个合并的DataFrame
    - False: 包含各文件DataFrame的列表
    
    异常:
    ValueError - 当write_csv=True但未提供output_dir时
    """
    if write_csv and not output_dir:
        raise ValueError("write_csv为True时必须指定output_dir")
    
    # 获取目录下所有json文件
    json_files = [
        f for f in os.listdir(json_dir) 
        if f.endswith('.json') and os.path.isfile(os.path.join(json_dir, f))
    ]
    
    if not json_files:
        return pd.DataFrame() if combine else []
    
    results = []
    current_id = start_id
    
    for json_file in json_files:
        try:
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # 转换当前文件
            df = convert_anylabeling_to_pd(
                json_data=data,
                rename_label=rename_label,
                current_id=current_id
            )
            
            if not df.empty:
                # 添加来源文件名（不带扩展名）
                df['source_file'] = os.path.splitext(json_file)[0]
                
                if combine:
                    current_id += len(df)
                
                if write_csv:
                    output_path = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(json_file)[0]}.csv"
                    )
                    df.to_csv(output_path, index=False)
                
                results.append(df)
                
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            continue
    
    if combine:
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    else:
        return results



if __name__ == "__main__":
    import json
#    from deepforest.visualize import plot_results
#   from deepforest import utilities
    processed_dfs = process_json_files(
    json_dir="./test_img",
    start_id=1,
    combine=True,
    write_csv=False,
    output_dir="./test_img"
)
    print(processed_dfs)
#   converted_df = utilities.read_file("./test_img/crop_637.csv")   
#  plot_results(converted_df)
# plot_results无法识别包含polygon的csv文件，可能需要其他能识别的格式