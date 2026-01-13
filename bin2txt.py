import numpy as np

def bin_to_txt_correctly(bin_file_path, txt_file_path):
    try:
        # NumPy 배열로 불러오기
        data = np.load(bin_file_path)
        breakpoint()
        # 텍스트 파일로 저장
        np.savetxt(txt_file_path, data, fmt='%.8f')  # 소수점 8자리로 변환
        print(f"Successfully converted {bin_file_path} to {txt_file_path}")

    except Exception as e:
        print(f"Error: {e}")

# 파일 경로 지정
bin_file = 'data/semi/train/raw/agv/01/agv01_0901_0812/agv01_0901_081240.bin'
txt_file = 'data/semi/train/raw/agv/01/agv01_0901_0812/agv01_0901_081240.txt'

# 변환 실행
bin_to_txt_correctly(bin_file, txt_file)
