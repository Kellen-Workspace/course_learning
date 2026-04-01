import numpy as np

def read_envi_dn_to_radiance_simple(filename, nl=557, ns=530, nb=1,
                                    radiance_mult=0.0003342, radiance_add=0.1):
    """
    简化版：读取ENVI数据并定标
    """
    import rasterio

    # 方法1: 使用rasterio
    with rasterio.open(filename) as src:
        dn_data = src.read().astype(np.float32)

    # 如果是多波段，取第一个波段
    if dn_data.ndim == 3 and dn_data.shape[0] > 1:
        dn_data = dn_data[0, :, :]
        # 辐射定标
    print(dn_data.shape)
    radiance_data = dn_data * radiance_mult + radiance_add

    return radiance_data[0, :, :]


def read_envi_lse(filename, nl=557, ns=530, nb=1,):
    """
    简化版：读取ENVI数据并定标
    """
    import rasterio

    # 方法1: 使用rasterio
    with rasterio.open(filename) as src:
        dn_data = src.read().astype(np.float32)

    # 如果是多波段，取第一个波段
    if dn_data.ndim == 3 and dn_data.shape[0] > 1:
        dn_data = dn_data[0, :, :]
        # 辐射定标
    print(dn_data.shape)

    return dn_data[0, :, :]

def cal_LST(radiance,lse,radiance_up,radiance_down,trans_rate):
    radiance_surface=(radiance-radiance_up-(1-lse)*radiance_down*trans_rate)/lse/trans_rate
    Ts=1321.08/np.log(774.89/radiance_surface+1)
    return Ts

if __name__ == '__main__':
    radiance=read_envi_dn_to_radiance_simple(r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\Data\DN_image.envi")
    lse=read_envi_lse(r"D:\BaiduNetdiskDownload\热红外上机课件-20251201\【7】上机课件-20251201\Data\LSE.envi")

    LST=cal_LST(radiance,lse,1.395671,2.226653,0.784500)
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 最简化版本 - 一图两用
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左侧：温度图
    im = ax1.imshow(LST, cmap='jet')
    plt.colorbar(im, ax=ax1, label='温度 (K)')
    ax1.set_title('地表温度分布')
    ax1.axis('off')

    # 右侧：直方图
    ax2.hist(LST.flatten(), bins=200, color='red', alpha=0.7)
    ax2.set_xlabel('温度 (K)')
    ax2.set_ylabel('像素数量')
    ax2.set_title('温度分布直方图')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


