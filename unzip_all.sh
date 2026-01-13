#!/bin/bash

# 사용법 안내
if [ $# -eq 0 ]; then
    echo "사용법: $0 <디렉토리명>"
    echo "예시: $0 /path/to/directory"
    exit 1
fi

# 입력받은 디렉토리 경로
TARGET_DIR="$1"

# 디렉토리가 존재하는지 확인
if [ ! -d "$TARGET_DIR" ]; then
    echo "오류: 디렉토리 '$TARGET_DIR'가 존재하지 않습니다."
    exit 1
fi

# 디렉토리로 이동
cd "$TARGET_DIR"

# zip 파일이 있는지 확인
if ! ls *.zip 1> /dev/null 2>&1; then
    echo "오류: '$TARGET_DIR' 디렉토리에 zip 파일이 없습니다."
    exit 1
fi

echo "디렉토리 '$TARGET_DIR'에서 zip 파일들을 압축해제합니다..."

# 각 zip 파일을 처리
for zipfile in *.zip; do
    # .zip 확장자를 제거한 디렉토리명 생성
    dirname="${zipfile%.zip}"
    
    echo "압축해제 중: $zipfile -> $dirname/"
    
    # 디렉토리가 이미 존재하는지 확인
    if [ -d "$dirname" ]; then
        echo "경고: 디렉토리 '$dirname'가 이미 존재합니다. 덮어씁니다."
        rm -rf "$dirname"
    fi
    
    # 디렉토리 생성 및 압축해제
    mkdir -p "$dirname"
    if unzip -q "$zipfile" -d "$dirname"; then
        echo "성공: $zipfile -> $dirname/"
    else
        echo "오류: $zipfile 압축해제 실패"
        # 실패한 경우 생성된 디렉토리 삭제
        rm -rf "$dirname"
    fi
done

echo "모든 zip 파일 압축해제 완료!" 