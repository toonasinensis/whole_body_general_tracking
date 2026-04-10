#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将npz文件中的数据提取并保存为CSV格式
格式: body_pos_x, body_pos_y, body_pos_z, body_vel_x, body_vel_y, body_vel_z,
      body_quat_w, body_quat_x, body_quat_y, body_quat_z, 
      joint_00-joint_20, joint_vel_00-joint_vel_20
"""

import numpy as np
import argparse
import os


def npz_to_csv(npz_path, output_path=None):
    """
    将npz文件转换为CSV格式
    
    Args:
        npz_path: npz文件路径
        output_path: 输出CSV文件路径，如果为None则自动生成
    """
    # 加载npz文件
    data = np.load(npz_path)
    
    print(f"NPZ文件包含的键: {list(data.keys())}")
    
    # 打印每个键的形状以便调试
    for key in data.keys():
        print(f"  {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
    
    # 根据实际数据结构提取信息
    # 数据结构可能是：
    # 情况1: 'body_pos_w': (T, N, 3), 'body_quat_w': (T, N, 4) - 多个身体部位
    # 情况2: 'body_pos': (T, 3), 'body_quat': (T, 4) - 单个根部
    # 关节数据: 'joint_pos' 或 'dof_pos': (T, 21)
    
    # 尝试识别数据字段
    body_pos = None
    body_quat = None
    joint_angles = None
    joint_vel = None
    
    # 查找身体位置和四元数（处理多身体部位的情况）
    if 'body_pos_w' in data and len(data['body_pos_w'].shape) == 3:
        # 提取根部（索引0）的位置
        body_pos = data['body_pos_w'][:, 0, :]
        print(f"  从 'body_pos_w' 提取根部位置: {body_pos.shape}")
    elif 'body_pos' in data:
        body_pos = data['body_pos']
        if len(body_pos.shape) == 3:
            body_pos = body_pos[:, 0, :]
    elif 'root_pos' in data:
        body_pos = data['root_pos']
    
    if 'body_quat_w' in data and len(data['body_quat_w'].shape) == 3:
        # 提取根部（索引0）的四元数
        body_quat = data['body_quat_w'][:, 0, :]
        print(f"  从 'body_quat_w' 提取根部四元数: {body_quat.shape}")
    elif 'body_quat' in data:
        body_quat = data['body_quat']
        if len(body_quat.shape) == 3:
            body_quat = body_quat[:, 0, :]
    elif 'root_quat' in data:
        body_quat = data['root_quat']
    
    # 查找关节角度
    if 'joint_pos' in data:
        joint_angles = data['joint_pos']
        print(f"  识别 'joint_pos' 为关节数据: {joint_angles.shape}")
    elif 'joint_angles' in data:
        joint_angles = data['joint_angles']
    elif 'dof_pos' in data:
        joint_angles = data['dof_pos']
    elif 'joints' in data:
        joint_angles = data['joints']
    
    # 查找关节速度
    if 'joint_vel' in data:
        joint_vel = data['joint_vel']
        print(f"  识别 'joint_vel' 为关节速度数据: {joint_vel.shape}")
    elif 'dof_vel' in data:
        joint_vel = data['dof_vel']
    
    # 如果找不到，尝试从所有数据中推断
    if body_pos is None or body_quat is None or joint_angles is None:
        print("\n尝试自动识别数据结构...")
        for key in data.keys():
            arr = data[key]
            if len(arr.shape) == 2:
                if arr.shape[1] == 3 and body_pos is None:
                    body_pos = arr
                    print(f"  识别 '{key}' 为 body_pos")
                elif arr.shape[1] == 4 and body_quat is None:
                    body_quat = arr
                    print(f"  识别 '{key}' 为 body_quat")
                elif arr.shape[1] == 21 and joint_angles is None:
                    joint_angles = arr
                    print(f"  识别 '{key}' 为 joint_angles")
    
    # 检查是否所有数据都找到了
    if body_pos is None:
        print("错误: 找不到身体位置数据 (期望形状: (T, 3))")
        return
    if body_quat is None:
        print("错误: 找不到身体四元数数据 (期望形状: (T, 4))")
        return
    if joint_angles is None:
        print("错误: 找不到关节角度数据 (期望形状: (T, 27))")
        return
    if joint_vel is None:
        print("警告: 找不到关节速度数据，将使用零值填充")
        joint_vel = np.zeros_like(joint_angles)
    
    print(f"\n数据形状:")
    print(f"  body_pos: {body_pos.shape}")
    print(f"  body_quat: {body_quat.shape}")
    print(f"  joint_angles: {joint_angles.shape}")
    print(f"  joint_vel: {joint_vel.shape}")
    
    # 确保所有数据的帧数一致
    num_frames = body_pos.shape[0]
    all_shapes = [body_quat.shape[0], joint_angles.shape[0], joint_vel.shape[0]]
    if any(s != num_frames for s in all_shapes):
        print(f"警告: 数据帧数不一致!")
        num_frames = min(num_frames, *all_shapes)
        print(f"使用最小帧数: {num_frames}")
    
    # 合并数据
    # body_pos: (T, 3)
    # body_quat: (T, 4) - 需要确保是 wxyz 格式
    # joint_angles: (T, 27)
    # joint_vel: (T, 27)
    combined_data = np.concatenate([
        body_pos[:num_frames],
        body_quat[:num_frames],
        joint_angles[:num_frames],
        joint_vel[:num_frames]
    ], axis=1)
    
    print(f"\n合并后的数据形状: {combined_data.shape} (期望: ({num_frames}, 61))")
    
    # 生成列名
    header = ['body_pos_x', 'body_pos_y', 'body_pos_z',
              'body_quat_w', 'body_quat_x', 'body_quat_y', 'body_quat_z']
    header += [f'joint_pos{i:02d}' for i in range(27)]
    header += [f'joint_vel_{i:02d}' for i in range(27)]
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(npz_path)[0]
        output_path = f"{base_name}.csv"
    
    # 保存为CSV
    np.savetxt(output_path, combined_data, delimiter='\t', 
               header='\t'.join(header), comments='', fmt='%.6f')
    
    print(f"\n成功保存到: {output_path}")
    print(f"总帧数: {num_frames}")
    print(f"总列数: {len(header)}")


def main():
    parser = argparse.ArgumentParser(description='将NPZ文件转换为CSV格式')
    parser.add_argument('input', type=str, help='输入的NPZ文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='输出的CSV文件路径（默认与输入文件同名）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        return
    
    npz_to_csv(args.input, args.output)


if __name__ == '__main__':
    main()
