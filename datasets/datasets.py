import re


def is_valid_line(line):
    # 去掉首尾空白字符
    line = line.strip()
    # 检查行是否不超过12个字符
    if len(line) <= 12:
        return False
    # 检查行是否包含非汉字非标点符号的字符
    if re.search(r'[^\u4e00-\u9fa5，。！？；：、]', line):
        return False
    return True


def process_line(line):
    # 删除冒号及冒号之前的文本
    if '：' in line:
        line = line.split('：', 1)[-1]
    return line.strip()


def filter_and_merge_lines(input_file, output_file):
    with open(input_file, 'r',
              encoding='utf-8') as infile, open(output_file,
                                                'w',
                                                encoding='utf-8') as outfile:
        buffer = ""
        for line in infile:
            line = process_line(line)
            if is_valid_line(line):
                if buffer:
                    buffer += line
                    if buffer.endswith('。'):
                        outfile.write(buffer + '\n')
                        buffer = ""
                else:
                    if line.endswith('。'):
                        outfile.write(line + '\n')
                    else:
                        buffer = line
        # 将最后一个缓冲区写入，如果不以句号结尾，添加句号
        if buffer:
            if not buffer.endswith('。'):
                buffer += '。'
            outfile.write(buffer + '\n')


# 调用示例
input_file = 'songci.txt'
output_file = 'datasets.txt'
filter_and_merge_lines(input_file, output_file)
