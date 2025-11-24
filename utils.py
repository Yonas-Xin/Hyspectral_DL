import numpy as np
import os
import torch
import matplotlib.colors as mcolors
from datetime import datetime

def get_full_academic_color_256():
    FULL_ACADEMIC_COLOR_256 = [
    '#D5E5C9', '#1C6995', '#D9C2DF', '#E2795A', '#EAC56C', '#299D90', '#895C56', '#1BB5B9',
    '#D68E04', '#EEA78B', '#459741', '#9566A8', '#A4D2A1', '#E98D49', '#639DFC', '#93A906',
    '#C0FDD8', '#FEC0C1', '#CDC6FF', '#FDC0F7', '#F3D8F1', '#D6EBBF', '#E1CAF7', '#BFDCE2',
    '#F8F0BE', '#BEEFBF', '#F8C9C8', '#C0E2D2', '#E9BFC0', '#E3E3E3', '#BFBFBF', '#DEECF6',
    '#AFCBE2', '#E2F2CD', '#B6DAA7', '#F9D5D5', '#EF9BA1', '#FBE3C0', '#FBC99A', '#EBE0EF',
    '#C2B1D7', '#6899E6', '#CF66A6', '#66CC88', '#996B66', '#9366CC', '#CC9C66', '#66B0CC',
    '#D96672', '#66D96E', '#7D6699', '#CC8D66', '#66BACC', '#D966A4', '#75D966', '#6B6699',
    '#CC7D66', '#66C4CC', '#D966D6', '#A7D966', '#666E99', '#CC6E66', '#66CCCC', '#C866D9',
    '#D9D966', '#668099', '#B4CC66', '#66CCB6', '#9666D9', '#D9A766', '#669199', '#C4CC66',
    '#66CCA2', '#666BD9', '#D97566', '#66A399', '#D3CC66', '#66CC8E', '#669DD9', '#D9668D',
    '#66B499', '#CCBA66', '#66CC7A', '#66CFD9', '#C666D9', '#66C699', '#CCAA66', '#66CC66',
    '#669DD9', '#9466D9', '#77CC99', '#CC9B66', '#66CC70', '#66B9D9', '#66D9A4', '#89CC99',
    '#CC8B66', '#66CC7A', '#66D5D9', '#66D973', '#9ACC99', '#CC7C66', '#66CC85', '#66F1D9',
    '#7DD966', '#ACCC99', '#CC6C66', '#66CC8F', '#82D9F1', '#AFD966', '#BDCC99', '#BA66CC',
    '#66CC99', '#9ED9F1', '#E1D966', '#CECC99', '#8866CC', '#66CCA3', '#B9D9F1', '#D99D66',
    '#DFCC99', '#6677CC', '#66CCAD', '#D5D9F1', '#D96B66', '#F1CC99', '#66A9CC', '#66CCB7',
    '#F1D9F1', '#D96698', '#E6CCAD', '#66DACC', '#66CCC2', '#F1D9D5', '#C166D9', '#D7CCAD',
    '#66CCD6', '#66CCCD', '#D5F1D9', '#8F66D9', '#C8CCAD', '#66C0CC', '#66CCD7', '#B9F1D9',
    '#666AD9', '#B9CCAD', '#66B5CC', '#66CCE1', '#9EF1D9', '#669CD9', '#AACCAD', '#66AACC',
    '#66CCEB', '#82F1D9', '#66CED9', '#9BCCAD', '#669FCC', '#66CCF5', '#78F1D9', '#8CCCAD',
    '#6693CC', '#66CCFF', '#66D9BE', '#94F1D9', '#7DCCAD', '#6688CC', '#66C2FF', '#66D98C',
    '#B0F1D9', '#6ECCAD', '#667DCC', '#66B8FF', '#CCF1D9', '#66CCB2', '#6672CC', '#66ADFF',
    '#E8F1D9', '#66CC9C', '#6667CC', '#66A3FF', '#F1E8D9', '#66CC86', '#7866CC', '#6698FF',
    '#F1CCD9', '#66CC70', '#AA66CC', '#668EFF', '#F1B0D9', '#66CC66', '#DC66CC', '#6683FF',
    '#F194D9', '#6FCC66', '#F166CC', '#6679FF', '#F178D9', '#85CC66', '#CC66DD', '#666EFF',
    '#B1CC66', '#6A66DD', '#6B6EFF', '#C7CC66', '#666ADD', '#876EFF', '#DDCC66', '#669BDD',
    '#A36EFF', '#DDC166', '#66CCDD', '#BF6EFF', '#DDB666', '#DB6EFF', '#D99D66', '#DDAB66',
    '#66DDCD', '#F76EFF', '#DDA066', '#66DDB2', '#F186FF', '#DD9566', '#66DD97', '#D586FF',
    '#F2D9D5', '#F1CE66', '#8A2BE2', '#FF4500', '#2E8B57', '#DA70D6', '#FF6347', '#40E0D0',
    '#FF69B4', '#9ACD32', '#FFA500', '#BA55D3', '#00CED1', '#FFB6C1', '#7B68EE', '#00FA9A',
    '#DDA0DD', '#FFD700', '#ADFF2F', '#FF00FF', '#1E90FF', '#F0E68C', '#E6E6FA', '#98FB98',
    '#D3D3D3', '#FFDEAD', '#F5F5DC', '#F08080', '#AFEEEE', '#D8BFD8', '#B0C4DE', '#FFFACD',
    '#E0FFFF', '#FAFAD2', '#FFE4E1', '#F0FFF0', '#F5F5F5', '#FFF0F5', '#F8F8FF', '#F0F8FF',
    '#F5DEB3', '#DEB887', '#5F9EA0', '#8470FF', '#778899', '#B8860B', '#A9A9A9', '#006400',
    '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F',
    '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#FF00FF',
    '#FFDAB9', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00',
    '#FFF8DC', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
    '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B',
    '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#CAFF70', '#D3FF70', '#BCFF70'
    ]
    # 前16个颜色为无重复的学术颜色，其余只是充数的颜色
    return FULL_ACADEMIC_COLOR_256

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_epochs, num_batches, meters, prefix: str = "") -> None:
        self.epoch_fmtstr = self._get_batch_fmtstr(num_epochs)
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = f'{formatted_time} ' + "\t".join(entries)
        print(out)
    
    def epoch_summary(self, epoch, other_str=""):
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = ["Epoch" + self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        result = f"{formatted_time} "+"\t".join(entries)+"\t"+other_str
        print(result)
        return result

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def topk_accuracy(output, target, topk=(1,)): # 计算Top-k准确率
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def block_generator(data, block_size=256):
    '''迭代器，输入一个影像，返回分块的位置掩膜'''
    if data.ndim == 3:
        rows, cols, _ = data.shape
    else: rows, cols = data.shape
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # 计算当前块的实际高度和宽度（避免越界）
            actual_rows = min(block_size, rows - i)
            actual_cols = min(block_size, cols - j)
            position_mask = np.zeros((rows, cols))
            position_mask[i:i+actual_rows, j:j+actual_cols] = 1
            yield position_mask

def label_to_rgb(t, MAP=get_full_academic_color_256(), background_value=-1):
    '''根据颜色条将label映射到rgb图像'''
    MAP = batch_hex_to_rgb(MAP) # 将HEX值转化为rgb值
    H, W = t.shape
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    mask = t != background_value # 确定背景值
    rgb[mask] = np.array(MAP)[t[mask]]
    rgb[~mask] = [255, 255, 255]
    return rgb

def hex_to_rgb(value):
    rgb = mcolors.hex2color(value)
    rgb_255 = [int(round(x * 255)) for x in rgb]
    return rgb_255

def batch_hex_to_rgb(value_list : list):
    return [hex_to_rgb(i) for i in value_list]

def search_files_in_directory(directory, extension):
    """
    搜索指定文件夹中所有指定后缀名的文件，并返回文件路径列表,只适用于不需要标签训练的模型，因为返回的列表顺序可能和
    需要的顺序不同，使用需慎重，但是同一命名规则返回的列表一定是相同的
    Parameters:
        directory (str): 要搜索的文件夹路径
        extension (str): 文件后缀名，应该以 '.' 开头，例如 '.txt', '.jpg'
    Returns:
        list: 包含所有符合条件的文件路径的列表
    """
    matching_files = []
    if isinstance(extension, str):
        extension = (extension,)
    elif isinstance(extension, tuple):
        pass
    else:
        raise ValueError("The suffix must be a string or a tuple!")
    for root, dirs, files in os.walk(directory):
        for file in files:
            for ex in extension:
                if file.endswith(ex):
                    matching_files.append(os.path.join(root, file))
    return matching_files

def read_txt_to_list(filename):
    with open(filename, 'r') as file:
        # 逐行读取文件并去除末尾的换行符
        data = [line.strip() for line in file.readlines()]
    return data

def write_list_to_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # 每个元素后加上换行符
        file.flush()
        
def read_dataset_from_txt(txt_file):
    'txt文件绝对地址'
    parent_dir = os.path.dirname(txt_file)
    paths = read_txt_to_list(txt_file)
    x = [os.path.basename(i) for i in paths]
    y = [os.path.join(parent_dir, i) for i in x]
    return y

def save_matrix_to_csv(matrix, filename, delimiter=','):
    """
    将 NumPy 矩阵或 PyTorch 张量（二维）保存为 CSV 文件
    
    参数:
        matrix: 输入的 NumPy 矩阵或 PyTorch 张量（二维）
        filename: 要保存的 CSV 文件名（包括路径）
        delimiter: CSV 分隔符，默认为 ','
    
    返回:
        None
    """
    if not isinstance(matrix, (np.ndarray, torch.Tensor)):
        raise ValueError("输入必须是 NumPy 数组或 PyTorch 张量")
    if len(matrix.shape) > 2:
        raise ValueError("输入必须是二维矩阵")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy() # 转为 NumPy 数组
    if matrix.dtype == np.bool: # bool 类型转换为 int
        matrix = matrix.astype(np.int16)
    # 使用 NumPy 保存为 CSV
    np.savetxt(filename, matrix, delimiter=delimiter, fmt='%s')
    print(f"data has been saved as csv: {filename}")

def read_csv_to_matrix(filename, delimiter=','):
    """
    从 CSV 文件读取数据并转换为 NumPy 矩阵
    
    参数:
        filename: 要读取的 CSV 文件名（包括路径）
        delimiter: CSV 分隔符，默认为 ','
    
    返回:
        NumPy 矩阵
    """
    return np.loadtxt(filename, delimiter=delimiter, dtype=np.float32)

if __name__ == '__main__':
    # 测试颜色转换函数
    color = get_full_academic_color_256()
    print(len(color))