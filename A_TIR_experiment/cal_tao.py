import numpy as np
from scipy import io
from load_tp7_file import read_tp7_file


def read_tirs_srf(mat_file):
    """读取Landsat TIRS通道响应函数"""
    data = io.loadmat(mat_file)
    srf_data = data['srf_tirs']

    wavenumber = srf_data[:, 0]  # 波数 (cm⁻¹)
    srf_b10 = srf_data[:, 1]  # 波段10响应函数
    srf_b11 = srf_data[:, 2]  # 波段11响应函数

    wavelength = 10000 / wavenumber  # 波数转波长

    return wavenumber, srf_b10, srf_b11, wavelength


import matplotlib.pyplot as plt


def plot_transmittance_with_srf_single(wavenumber, transmittance,
                                       srf_b10, srf_b11,
                                       trans_b10=None, trans_b11=None):
    """
    在同一张图上绘制透过率与两个通道响应函数（基于波长）
    假设srf_wavenumber和tp7_wavenumber相同

    参数:
    wavenumber: 波数数组 (cm⁻¹)，同时用于透过率和响应函数
    transmittance: 透过率数组
    srf_b10: B10通道响应函数数组
    srf_b11: B11通道响应函数数组
    trans_b10: B10通道平均透过率（可选）
    trans_b11: B11通道平均透过率（可选）
    """

    # 计算波长：λ(μm) = 10000 / ν(cm⁻¹)
    wavelength = 10000 / wavenumber

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 创建双y轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # 左y轴：透过率（黑色实线）
    line1, = ax1.plot(wavelength, transmittance, 'k-',
                      linewidth=2.5, label='大气透过率')
    ax1.set_xlabel('波长 (μm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('透过率', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 1.1)

    # 右y轴：响应函数（彩色虚线）
    line2, = ax2.plot(wavelength, srf_b10, 'b--',
                      linewidth=2, alpha=0.8, label='B10响应函数')
    line3, = ax2.plot(wavelength, srf_b11, 'r--',
                      linewidth=2, alpha=0.8, label='B11响应函数')
    ax2.set_ylabel('响应函数', fontsize=12, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1.1)

    # 标记通道中心波长
    ax1.axvline(x=10.9, color='b', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axvline(x=12.0, color='r', linestyle=':', alpha=0.5, linewidth=1)

    # 添加中心波长文字标注
    ax1.text(10.9, 1.05, '10.9 μm (B10)', ha='center',
             fontsize=10, color='blue', alpha=0.8)
    ax1.text(12.0, 1.05, '12.0 μm (B11)', ha='center',
             fontsize=10, color='red', alpha=0.8)

    # 如果提供了通道平均透过率，在图上标记
    if trans_b10 is not None and trans_b11 is not None:
        # 找到B10中心波长的索引
        b10_wavelength = 10.9
        idx_b10 = np.argmin(np.abs(wavelength - b10_wavelength))

        # 找到B11中心波长的索引
        b11_wavelength = 12.0
        idx_b11 = np.argmin(np.abs(wavelength - b11_wavelength))

        # 绘制标记点
        ax1.plot(b10_wavelength, trans_b10, 'bo',
                 markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax1.plot(b11_wavelength, trans_b11, 'ro',
                 markersize=10, markerfacecolor='white', markeredgewidth=2)

        # 添加文本说明（通道平均透过率）
        ax1.text(b10_wavelength, trans_b10 + 0.02, f'{trans_b10:.4f}',
                 ha='center', va='bottom', fontsize=11,
                 color='blue', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                            facecolor='white',
                                                            alpha=0.8))
        ax1.text(b11_wavelength, trans_b11 + 0.02, f'{trans_b11:.4f}',
                 ha='center', va='bottom', fontsize=11,
                 color='red', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                           facecolor='white',
                                                           alpha=0.8))

    # 合并图例
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

    # 设置标题
    plt.title('大气透过率与TIRS通道响应函数', fontsize=14, fontweight='bold', pad=20)

    # 设置x轴范围（TIRS主要工作范围）
    ax1.set_xlim(8, 14)



    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def conver_radian_UpDown(radiance_up, radiance_down, srf_wavenumber, srf_b10, srf_b11):
    """
    计算通道上下行辐射的加权平均值（单位转换）

    参数:
    radiance_up: 大气上行辐射 (W cm^-2 sr^-1 cm)
    radiance_down: 大气下行辐射 (W cm^-2 sr^-1 cm)
    srf_wavenumber: 响应函数波数 (cm⁻¹)
    srf_b10: B10通道响应函数
    srf_b11: B11通道响应函数

    返回:
    rad_up_tirs1, rad_up_tirs2, rad_down_tirs1, rad_down_tirs2: 通道加权辐射值
    """

    # 1. 转换单位: W cm^-2 sr^-1 cm → W m^-2 sr^-1 cm
    rad_up_wn = radiance_up * 10000
    rad_down_wn = radiance_down * 10000

    # 2. 转换单位: W m^-2 sr^-1 cm → W m^-2 sr^-1 um^-1
    # 公式: rad_wl = rad_wn * wn^2 / 10000
    rad_up_wl = rad_up_wn * srf_wavenumber ** 2 / 10000
    rad_down_wl = rad_down_wn * srf_wavenumber ** 2 / 10000

    # 3. 计算通道加权平均辐射值
    rad_up_tirs1 = np.sum(srf_b10 * rad_up_wl) / np.sum(srf_b10)
    rad_up_tirs2 = np.sum(srf_b11 * rad_up_wl) / np.sum(srf_b11)
    rad_down_tirs1 = np.sum(srf_b10 * rad_down_wl) / np.sum(srf_b10)
    rad_down_tirs2 = np.sum(srf_b11 * rad_down_wl) / np.sum(srf_b11)

    return rad_up_tirs1, rad_up_tirs2, rad_down_tirs1, rad_down_tirs2, rad_up_wl, rad_down_wl


def plot_radiance_up_down(wavelength, rad_up_wl, rad_down_wl,
                          rad_up_tirs1, rad_up_tirs2,
                          rad_down_tirs1, rad_down_tirs2):
    """
    绘制上下行辐射曲线图

    参数:
    wavelength: 波长数组 (μm)
    rad_up_wl: 上行辐射 (W m^-2 sr^-1 μm^-1)
    rad_down_wl: 下行辐射 (W m^-2 sr^-1 μm^-1)
    rad_up_tirs1: B10上行通道值
    rad_up_tirs2: B11上行通道值
    rad_down_tirs1: B10下行通道值
    rad_down_tirs2: B11下行通道值
    """

    # 通道中心波长
    wl_tirs1 = 10.8  # B10中心波长
    wl_tirs2 = 12.0  # B11中心波长

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 子图1: 上行辐射
    plt.subplot(2, 1, 1)
    plt.plot(wavelength, rad_up_wl, 'k-', linewidth=1.5, label='rad-up spectral')

    # 标记通道值
    plt.plot([wl_tirs1, wl_tirs2], [rad_up_tirs1, rad_up_tirs2],
             'mo-', markersize=8, linewidth=2, markerfacecolor='white',
             label='rad-up band')

    plt.xlabel('wavelength(um)', fontsize=12)
    plt.ylabel('radiance upwelling (W m⁻² sr⁻¹ μm⁻¹)', fontsize=12)
    plt.title('Radiance Upwelling - wavelength', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 14)

    # 添加通道值标注
    plt.text(wl_tirs1, rad_up_tirs1 * 1.05, f'{rad_up_tirs1:.4f}',
             ha='center', va='bottom', fontsize=10, color='m')
    plt.text(wl_tirs2, rad_up_tirs2 * 1.05, f'{rad_up_tirs2:.4f}',
             ha='center', va='bottom', fontsize=10, color='m')

    # 子图2: 下行辐射
    plt.subplot(2, 1, 2)
    plt.plot(wavelength, rad_down_wl, 'k-', linewidth=1.5, label='rad-down spectral')

    # 标记通道值
    plt.plot([wl_tirs1, wl_tirs2], [rad_down_tirs1, rad_down_tirs2],
             'mo-', markersize=8, linewidth=2, markerfacecolor='white',
             label='rad-down band')

    plt.xlabel('wavelength (um)', fontsize=12)
    plt.ylabel('radiance downwelling (W m⁻² sr⁻¹ μm⁻¹)', fontsize=12)
    plt.title('Radiance Downwelling - wavelength', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 14)

    # 添加通道值标注
    plt.text(wl_tirs1, rad_down_tirs1 * 1.05, f'{rad_down_tirs1:.4f}',
             ha='center', va='bottom', fontsize=10, color='m')
    plt.text(wl_tirs2, rad_down_tirs2 * 1.05, f'{rad_down_tirs2:.4f}',
             ha='center', va='bottom', fontsize=10, color='m')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. 读取MODTRAN .tp7文件
    tp7_wavenumber, transmittance, radiance_up, radiance_down = read_tp7_file(
        r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\modtran\experiment.tp7"
    )

    # 2. 读取TIRS通道响应函数
    srf_wavenumber, srf_b10, srf_b11, wavelength = read_tirs_srf(r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\Data\srf_tirs.mat")


    trans_b10 = np.sum(transmittance * srf_b10) / np.sum(srf_b10)
    trans_b11 = np.sum(transmittance * srf_b11) / np.sum(srf_b11)
    print(tp7_wavenumber.shape)
    print(srf_wavenumber.shape)

    print("通道透过率计算结果:")
    print(f"B10 (10.9 μm) 透过率: {trans_b10:.6f}")
    print(f"B11 (12.0 μm) 透过率: {trans_b11:.6f}")

    #绘制透过率计算结果
    rad_up_tirs1, rad_up_tirs2, rad_down_tirs1, rad_down_tirs2, rad_up_wl, rad_down_wl = conver_radian_UpDown(
        radiance_up, radiance_down, srf_wavenumber, srf_b10, srf_b11
    )

    # 6. 绘制上下行辐射图
    #plot_radiance_up_down(wavelength, rad_up_wl, rad_down_wl,  rad_up_tirs1, rad_up_tirs2, rad_down_tirs1, rad_down_tirs2)

    print("辐射计算结果")
    print("=" * 50)
    print(f"B10 (10.8 μm):")
    print(f"  上行辐射: {rad_up_tirs1:.6f} W m⁻² sr⁻¹ μm⁻¹")
    print(f"  下行辐射: {rad_down_tirs1:.6f} W m⁻² sr⁻¹ μm⁻¹")
    print(f"B11 (12.0 μm):")
    print(f"  上行辐射: {rad_up_tirs2:.6f} W m⁻² sr⁻¹ μm⁻¹")
    print(f"  下行辐射: {rad_down_tirs2:.6f} W m⁻² sr⁻¹ μm⁻¹")




