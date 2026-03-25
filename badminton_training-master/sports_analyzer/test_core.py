# test_video_upload.py
import requests
import json


def test_video_upload(video_path):
    """测试视频上传功能"""
    url = "http://localhost:8000/api/v1/analysis/upload"

    try:
        with open(video_path, 'rb') as f:
            files = {'file': (video_path.split('/')[-1], f, 'video/mp4')}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            print("✅ 视频分析成功！")
            print(f"综合评分: {result.get('overall_score', 0)}")
            print(f"优势: {result.get('strengths', [])}")
            print(f"改进点: {result.get('weaknesses', [])}")
            return result
        else:
            print(f"❌ 分析失败: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ 请求失败: {e}")

    return None


# 测试
if __name__ == "__main__":
    test_video_upload("user1.mp4")  # 准备一个测试视频