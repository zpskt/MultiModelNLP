import requests

if __name__ == "__main__":
    # 定义输入文本列表
    pre_list = [
        '启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音',
        '我很喜欢',
        '吴总好帅',
        '海尔欠我的工钱到底什么时候才给我',
    ]

    # 发送 POST 请求
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": pre_list}  # 自动设置 Content-Type: application/json
    )

    # 输出响应结果
    print("Status Code:", response.status_code)
    print("Response:", response.json())
