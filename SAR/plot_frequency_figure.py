import numpy as np
import matplotlib.pyplot as plt

def plot_doppler_vs_time():
    # 参数定义 [根据作业要求]
    Vr = 7100          # 平台速度 (m/s)
    Rc = 850000        # 景中心斜距 (m)
    lam = 0.0556       # 波长 (m)
    Ta = 4          # 合成孔径时间 (s)

    # 生成方位时间轴 (慢时间)
    eta = np.linspace(-Ta/2, Ta/2, 1000)

    # 计算多普勒频率 (基于二阶展开推导)
    f_eta = -(2 * Vr**2 / (lam * Rc)) * eta

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(eta, f_eta, 'b', linewidth=2)
    plt.xlabel('Azimuth Time (s)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.grid(True)
    plt.show()


def plot_rd_2nd_order():
    # 参数定义 [基于图片给定参数]
    Vr = 7100          # 平台速度 (m/s)
    Rc = 850000        # 景中心斜距 (m)
    lam = 0.0556       # 波长 (m)
    Bdop = 8000        # 多普勒带宽 (Hz)

    # 生成多普勒频率轴
    f_eta = np.linspace(-Bdop/2, Bdop/2, 1000)

    # 距离-多普勒频率二阶近似公式
    R_f_eta = Rc + (lam**2 * Rc / (8 * Vr**2)) * f_eta**2

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(R_f_eta, f_eta, 'g', linewidth=2)
    plt.title('Range-Doppler Trajectory')
    plt.xlabel('Slant Range (m)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.grid(True)
    plt.show()


def plot_azimuth_phase_pi():
    # 参数设置
    Vr = 7100  # 平台速度 (m/s)
    Rc = 850000  # 景中心斜距 (m)
    lam = 0.0556  # 波长 (m)
    Ta = 3.33  # 合成孔径时间 (s)
    PRF = 8000 # 脉冲重复频率 (Hz)

    # 生成方位时间轴
    eta = np.arange(-Ta / 2, Ta / 2, 1 / PRF)

    # 计算原始相位 (二阶近似)
    phase = -(2 * np.pi * Vr ** 2 / (lam * Rc)) * eta ** 2

    # 将相位限制在 (-pi, pi]
    # 使用 np.angle(np.exp(1j * phase)) 是最标准的方法
    phase_wrapped = np.angle(np.exp(1j * phase))

    # 绘图
    plt.figure(figsize=(10, 5))
    # 将纵轴数值归一化到以 pi 为单位
    plt.plot(eta, phase_wrapped / np.pi, ',', color='blue')

    plt.xlabel('Azimuth Time (s)')
    plt.ylabel('Phase (rad)')  # 纵轴标签改为以 pi 为单位

    # 设置纵轴刻度显示
    plt.yticks([-1, -0.5, 0, 0.5, 1], ['-1$\pi$', '-0.5$\pi$', '0', '0.5$\pi$', '1$\pi$'])

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


plot_azimuth_phase_pi()