import numpy as np
import matplotlib.pyplot as plt
# 方法1：使用numpy.loadtxt（最常用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_tp7_file(filepath):

    data = np.loadtxt(filepath)

    # 提取各列（根据您的文档说明）
    wavenumber = data[:, 0]        # 第1列: 波数 (cm⁻¹)
    transmittance = data[:, 1]     # 第2列: 总透过率
    radiance_up = data[:, 2]            # 第3列: 大气上行辐射
    emis0 = 0.95  # 在.tp5文件中设置的SURREF=0.05，所以发射率=1-0.05=0.95
    ground_emis=data[:, 3]         # 第4列： 地表发射率
    ground_reflected = data[:, 4]  # 第5列： 地表反射辐射

    # 计算大气下行辐射
    radiance_down = ground_reflected / (1 - emis0) / transmittance

    return wavenumber,transmittance,radiance_up,radiance_down

if __name__ == '__main__':
    filepath=r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\modtran\example.tp7"
    read_tp7_file(filepath)
# # 绘制透过率曲线
# plt.figure(figsize=(10, 6))
# plt.plot(wavenumber, transmittance)
# plt.xlabel('Wavenumber (cm⁻¹)')
# plt.ylabel('Transmittance')
# plt.title('Atmospheric Transmittance from MODTRAN')
# plt.grid(True)
# plt.show()

# # 4. 绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(wavenumber, radiance_up, 'r-', linewidth=2, label='上行辐射 (Upwelling)')
# plt.plot(wavenumber, radiance_down, 'b-', linewidth=2, label='下行辐射 (Downwelling)')
# plt.xlabel('波数 (cm⁻¹)', fontsize=12)
# plt.ylabel('辐射亮度 (W·cm⁻²·sr⁻¹·cm⁻¹)', fontsize=12)
# plt.title('大气上行辐射和下行辐射波谱曲线', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, alpha=0.3)
# # 限制波数范围（可选，Landsat TIRS通常在800-1250 cm⁻¹）
# plt.xlim(800, 1250)
# plt.tight_layout()
# plt.show()

