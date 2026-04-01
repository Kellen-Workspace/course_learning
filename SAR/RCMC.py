import numpy as np
import matplotlib.pyplot as plt


def sar_optimized_processing():
    # --- 1. 参数设置 ---
    Vr, Rc, lam = 7100, 850000, 0.0556
    PRF, fs, c = 8000, 150e6, 3e8
    Ta = 0.6

    # 轴定义
    eta = np.arange(-Ta / 2, Ta / 2, 1 / PRF)
    N_az = len(eta)
    f_eta = np.fft.fftshift(np.fft.fftfreq(N_az, d=1 / PRF))

    # 距离向轴：严格按照 fs 定义以保证插值精度
    dr = c / (2 * fs)  # 采样间隔 (1m)
    dist_axis = np.arange(200) * dr - 30
    tau_axis = 2 * dist_axis / c

    # --- 2. 模拟回波 (时域生成，包含 RCM) ---
    data_time = np.zeros((len(dist_axis), N_az), dtype=complex)
    R_eta = Rc + (Vr ** 2 * eta ** 2) / (2 * Rc)

    for i in range(N_az):
        # 目标中心随时间移动 (RCM)
        target_pos = 2 * (R_eta[i] - Rc) / c
        # 距离向 Sinc + 方位向相位
        data_time[:, i] = np.sinc(fs * (tau_axis - target_pos)) * np.exp(-1j * 4 * np.pi * R_eta[i] / lam)

    # --- 3. 方位向 FFT 转到 RD 域 ---
    data_rd = np.fft.fftshift(np.fft.fft(data_time, axis=1), axes=1)

    # --- 4. RCMC (距离徙动校正) ---
    data_rcmc = np.zeros_like(data_rd)
    # 理论徙动量 Delta R = (lam^2 * Rc * f_eta^2) / (8 * Vr^2)
    delta_R = (lam ** 2 * Rc * f_eta ** 2) / (8 * Vr ** 2)
    delta_samples = delta_R / dr

    for j in range(N_az):
        old_idx = np.arange(len(dist_axis))
        # 使用线性插值将偏离位置映射回中心
        data_rcmc[:, j] = np.interp(old_idx, old_idx - delta_samples[j],
                                    data_rd[:, j], left=0, right=0)

    # --- 5. 方位向匹配滤波 (相位补偿) ---
    Ka = -2 * Vr ** 2 / (lam * Rc)
    Haz = np.exp(1j * np.pi * f_eta ** 2 / Ka)
    data_focused_rd = data_rcmc * Haz

    # --- 6. 方位向 IFFT 回到时域 ---
    data_final = np.fft.ifft(np.fft.ifftshift(data_focused_rd, axes=1), axis=1)

    # --- 7. 可视化输出 ---

    # 第一张图：RCMC 前后对比 (RD 域)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(data_rd), aspect='auto', extent=[f_eta[0], f_eta[-1], dist_axis[-1], dist_axis[0]])
    plt.title("Before RCMC")
    plt.xlabel("Doppler Frequency (Hz)");
    plt.ylabel("Relative Range (m)")

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(data_rcmc), aspect='auto', extent=[f_eta[0], f_eta[-1], dist_axis[-1], dist_axis[0]])
    plt.title("After RCMC")
    plt.xlabel("Doppler Frequency (Hz)");
    plt.ylabel("Relative Range (m)")
    plt.tight_layout()

    # 第二张图：最终聚焦结果与切面
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(data_final), aspect='auto', extent=[eta[0], eta[-1], dist_axis[-1], dist_axis[0]])
    plt.title("Final Focused Point Target")
    plt.xlabel("Azimuth Time (s)");
    plt.ylabel("Relative Range (m)")

    plt.subplot(1, 2, 2)
    # 提取目标中心行的方位向切面
    center_idx = np.argmin(np.abs(dist_axis))
    azimuth_profile = np.abs(data_final[center_idx, :])
    plt.plot(eta, azimuth_profile / np.max(azimuth_profile))
    plt.title("Azimuth Impulse Response (Sinc)")
    plt.xlabel("Azimuth Time (s)");
    plt.ylabel("Normalized Magnitude")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


sar_optimized_processing()