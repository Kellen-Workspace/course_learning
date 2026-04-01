import numpy as np
import rasterio
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [ 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def landsat8_radiometric_calibration(tif_file, metadata_file=None,
                                     radiance_mult=0.0003342,
                                     radiance_add=0.1):
    """
    Landsat 8辐射定标：DN值转辐亮度

    参数:
    tif_file: Landsat 8 TIF文件路径 (如B10.TIF)
    metadata_file: MTL元数据文件路径（可选，用于自动获取定标系数）
    radiance_mult: 定标乘性系数 (默认0.0003342)
    radiance_add: 定标加性系数 (默认0.1)

    返回:
    dn_data: DN值数组
    radiance_data: 辐亮度数组
    """

    # 1. 读取TIF文件
    with rasterio.open(tif_file) as src:
        dn_data = src.read(1)  # 读取第一个波段
        profile = src.profile  # 获取地理信息

    print(f"DN数据形状: {dn_data.shape}")
    print(f"DN值范围: {dn_data.min()} - {dn_data.max()}")

    # 2. 将DN值转换为float类型
    dn_data_float = dn_data.astype(np.float32)

    # 3. 处理填充值（0值设为NaN）
    dn_data_float[dn_data == 0] = np.nan

    # 4. 辐射定标：DN -> 辐亮度
    # 公式：Radiance = ML * DN + AL
    radiance_data = dn_data_float * radiance_mult + radiance_add

    print(f"辐亮度范围: {np.nanmin(radiance_data):.4f} - {np.nanmax(radiance_data):.4f} W·m⁻²·sr⁻¹·μm⁻¹")

    return dn_data, radiance_data, profile


def plot_dn_vs_radiance(dn_data, radiance_data):
    """绘制DN值和辐亮度的对比图"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. DN值图像
    im1 = axes[0].imshow(dn_data, cmap='viridis', vmin=np.nanpercentile(dn_data, 2),
                         vmax=np.nanpercentile(dn_data, 98))
    axes[0].set_title('DN值 (原始)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 辐亮度图像
    im2 = axes[1].imshow(radiance_data, cmap='hot',
                         vmin=np.nanpercentile(radiance_data, 2),
                         vmax=np.nanpercentile(radiance_data, 98))
    axes[1].set_title('辐亮度 (W·m⁻²·sr⁻¹·μm⁻¹)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def save_radiance_to_tif(radiance_data, profile, output_file):
    """保存辐亮度数据为TIF文件"""

    # 更新元数据
    profile.update(dtype=rasterio.float32, nodata=np.nan)

    # 保存为新的TIF文件
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(radiance_data.astype(np.float32), 1)

    print(f"辐亮度数据已保存到: {output_file}")


if __name__ == "__main__":

    landsat8_path_DN=r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\Data\LC81230322014279LGN00_B10.TIF"
    landsat8_path_radiance=r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\Data\LC8_B10_radiance.tif"

    # 方法1: 直接使用默认系数
    print("=== 方法1: 使用默认系数 ===")
    dn_data, radiance_data, profile = landsat8_radiometric_calibration(
        landsat8_path_DN
    )

    # 可视化
    print("\n=== 可视化结果 ===")
    plot_dn_vs_radiance(dn_data, radiance_data)

    # 保存结果
    print("\n=== 保存结果 ===")
    save_radiance_to_tif(radiance_data, profile, landsat8_path_radiance)

    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"DN值统计:")
    print(f"  均值: {np.nanmean(dn_data):.2f}")
    print(f"  标准差: {np.nanstd(dn_data):.2f}")
    print(f"  中位数: {np.nanmedian(dn_data):.2f}")

    print(f"\n辐亮度统计 (W·m⁻²·sr⁻¹·μm⁻¹):")
    print(f"  均值: {np.nanmean(radiance_data):.4f}")
    print(f"  标准差: {np.nanstd(radiance_data):.4f}")
    print(f"  中位数: {np.nanmedian(radiance_data):.4f}")